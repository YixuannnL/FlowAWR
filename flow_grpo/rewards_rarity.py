from PIL import Image
import os
import numpy as np
import torch
from collections import defaultdict


def aesthetic_score(device):
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def clip_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def hpsv2_score(device, capture_raw=False):
    from flow_grpo.hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        if capture_raw:
            # HPSv2.1 is used unscaled as the reward; the raw value is the scorer output itself.
            return scores, {"hpsv2_raw": scores.detach().float().cpu().numpy()}
        return scores, {}

    return _fn


def pickscore_score(device, capture_raw=False):
    from flow_grpo.pickscore_scorer_rarity import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        if capture_raw:
            scores, raw_scores, features = scorer(prompts, images, return_raw=True, return_features=True)
            # One batch transfer; diagnostic values stay out of reward arithmetic.
            return scores, {
                "pickscore_raw": raw_scores.detach().cpu().numpy(),
                "rarity_features": features.detach().cpu().numpy(),
            }
        scores, features = scorer(prompts, images, return_features=True)
        return scores, {"rarity_features": features.detach().cpu().numpy()}

    return _fn


def imagereward_score(device):
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def geneval_score(device, capture_raw=False):
    from flow_grpo.gen_eval import load_geneval

    batch_size = 64
    compute_geneval = load_geneval(device)

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        all_details = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            pil_images = [Image.fromarray(image) for image in image_batch]

            data = {
                "images": pil_images,
                "metadatas": list(metadata_batched),
                "only_strict": only_strict,
            }
            if capture_raw:
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards, results = compute_geneval(
                    **data, return_results=True
                )
                for result in results:
                    all_details.append(dict(
                        tag=result["tag"], score=float(result["score"]),
                        strict_correct=bool(result["strict_correct"]),
                        # evaluate() is skipped during training (only_strict=True); None marks "not computed".
                        correct=None if only_strict else bool(result["correct"]),
                        reason=result["reason"],
                    ))
            else:
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = compute_geneval(**data)

            all_scores += scores
            all_rewards += rewards
            all_strict_rewards += strict_rewards
            all_group_strict_rewards.append(group_strict_rewards)
            all_group_rewards.append(group_rewards)
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        reward_metadata = {}
        if capture_raw:
            # The training reward is the partial-credit score; keep it plus the per-sample verdicts.
            reward_metadata["geneval_raw"] = np.array(all_scores, dtype=np.float64)
            reward_metadata["geneval_detail"] = all_details
        return (
            all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict,
            reward_metadata,
        )

    return _fn


def ocr_score(device, capture_raw=False):
    from flow_grpo.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        if capture_raw:
            scores, details = scorer(images, prompts, return_raw=True)
            return scores, {"ocr_raw": np.array(scores, dtype=np.float64), "ocr_detail": details}
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn


def hpsv3_score(device):
    import pickle
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO

    batch_size = 64
    url = os.environ.get("REWARD_HPSV3_URL")
    if not url:
        raise ValueError("Set REWARD_HPSV3_URL to your HPSv3 reward service endpoint.")
    sess = requests.Session()
    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        pil_images = [Image.fromarray(image) for image in images]

        jpeg_images = []
        for img in pil_images:
            buf = BytesIO()
            img.save(buf, format="JPEG")
            jpeg_images.append(buf.getvalue())

        prompts = list(prompts)
        all_scores = []
        for i in range(0, len(jpeg_images), batch_size):
            data = {
                "images": jpeg_images[i : i + batch_size],
                "prompts": prompts[i : i + batch_size],
            }
            data_bytes = pickle.dumps(data)
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)
            all_scores += [float(x) for x in response_data["outputs"]]

        return all_scores, {}

    return _fn


def video_hpsv3(device):
    import cv2
    import math
    import pickle
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO

    batch_size = 64
    url = os.environ.get("REWARD_VIDEO_HPSV3_URL")
    if not url:
        raise ValueError("Set REWARD_VIDEO_HPSV3_URL to your video HPSv3 reward service endpoint.")
    sess = requests.Session()
    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _read_all_frames(video_path):
        """读取视频所有帧，返回 PIL.Image 列表"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()
        return frames

    def _fn(images, prompts, metadata):
        prompts = [m.get("prompt_v", m.get("prompt", "")) for m in metadata]
        if isinstance(images, (list, tuple)) and len(images) == 2 and isinstance(images[0], torch.Tensor):
            images = images[1]
        if not isinstance(images, (list, tuple)):
            images = [images]

        all_scores = []
        for item, prompt in zip(images, prompts):
            # 从 mp4 路径读取所有帧
            if isinstance(item, str):
                frames = _read_all_frames(item)
            else:
                # tensor fallback: [T, C, H, W] -> PIL
                if isinstance(item, torch.Tensor):
                    vid = (item * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                    if vid.ndim == 4 and vid.shape[1] == 3:
                        vid = vid.transpose(0, 2, 3, 1)  # [T,H,W,C]
                    frames = [Image.fromarray(f) for f in vid]
                else:
                    frames = []

            if not frames:
                all_scores.append(0.0)
                continue

            # 编码为 JPEG 字节
            jpeg_images = []
            for f in frames:
                buf = BytesIO()
                f.save(buf, format="JPEG")
                jpeg_images.append(buf.getvalue())

            data = {"images": jpeg_images, "prompts": [prompt] * len(jpeg_images)}
            data_bytes = pickle.dumps(data)
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            hps_scores = response_data["outputs"]
            l = len(hps_scores)
            hps_scores.sort(reverse=True)
            k = max(1, math.ceil(l * 0.3))
            video_score = sum(hps_scores[:k]) / k
            all_scores.append(video_score)

        return all_scores, {}

    return _fn


def videoalign_score(device):
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO 
    import pickle

    batch_size = 64
    url = os.environ.get("REWARD_VIDEOALIGN_SCORE_URL")
    if not url:
        raise ValueError("Set REWARD_VIDEOALIGN_SCORE_URL to your VideoAlign reward service endpoint.")
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(videos, prompts, metadata):
        if isinstance(videos, (list, tuple)) and len(videos) == 2 and isinstance(videos[0], torch.Tensor):
            videos = videos[1]
        if not isinstance(videos, (list, tuple)):
            videos = [videos]
        videos_batched = np.array_split(videos, np.ceil(len(videos) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(videos) / batch_size))

        all_scores = []
        for video_batch, prompt_batch in zip(videos_batched, prompts_batched):
            # format for LLaVA server
            data = {
                "path": video_batch.tolist(), 
                "prompts": prompt_batch.tolist(), 
            }
            # print(video_batch)

            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=600)
           
            response_data = pickle.loads(response.content)
            #response_data["videoalign_score"] = [float(x) for x in response_data["videoalign_score_mq"]]
            response_data["videoalign_score"] = [float(x['MQ']) for x in response_data["videoalign_score_all"]]

            # print(response_data["motion_scores"] )
            all_scores += response_data["videoalign_score"]
            # print(all_scores)

        return all_scores, {}

    return _fn 


# Rewards whose per-sample raw outputs can be captured for the raw-data logger.
RAW_CAPTURE_REWARDS = ("pickscore", "hpsv2", "ocr", "geneval")


def multi_score(device, score_dict, capture_raw=False):
    score_functions = {
        "ocr": ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "aesthetic": aesthetic_score,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "hpsv2": hpsv2_score,
        "hpsv3": hpsv3_score,
        "video_hpsv3": video_hpsv3,
        "videoalign_score": videoalign_score,
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        if capture_raw and score_name in RAW_CAPTURE_REWARDS:
            score_fns[score_name] = score_functions[score_name](device, capture_raw=True)
            continue
        score_fns[score_name] = (
            score_functions[score_name](device)
            if "device" in score_functions[score_name].__code__.co_varnames
            else score_functions[score_name]()
        )

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, only_strict=True):
        total_scores = None
        score_details = {}
        reward_metadata = {}

        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards, meta = score_fns[score_name](
                    images, prompts, metadata, only_strict
                )
                if capture_raw:
                    reward_metadata.update(meta)
                score_details["accuracy"] = rewards
                score_details["strict_accuracy"] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f"{key}_strict_accuracy"] = value
                for key, value in group_rewards.items():
                    score_details[f"{key}_accuracy"] = value
            else:
                scores, meta = score_fns[score_name](images, prompts, metadata)
                if capture_raw or score_name == "pickscore":
                    # Keep raw outputs in metadata, never in score_details/avg.
                    reward_metadata.update(meta)
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
            scores_arr = np.array(scores, dtype=np.float64)
            score_details[score_name] = scores_arr
            weighted = weight * scores_arr

            if total_scores is None:
                total_scores = weighted
            else:
                total_scores = total_scores + weighted

        score_details["avg"] = total_scores
        return score_details, reward_metadata

    return _fn
