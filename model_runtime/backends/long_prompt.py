"""77토큰보다 긴 SDXL 프롬프트를 75토큰 조각으로 나눠 인코딩한다.

CLIP 텍스트 인코더는 시작·끝 토큰을 빼고 75토큰까지만 읽는다.
그래서 그림체 문구가 긴 컨셉(조선·무협 등)에서 두 명이 나오는 컷은
장면 문장 끝(누가 무엇을 하는지)이 잘렸다. 조각마다 따로 인코딩해 이어 붙이면
뒤쪽 문구도 그림에 들어간다. 75토큰 이하 프롬프트는 이 경로를 쓰지 않아 예전과 픽셀이 같다.
조각은 되도록 쉼표 뒤에서 자른다("(old woman, gray hair)"가 두 조각으로 갈리지 않게).
"""

from __future__ import annotations

from typing import Any

# 시작·끝 토큰을 뺀 조각 길이
CHUNK = 75
# 쉼표를 찾아 되돌아가는 최대 토큰 수
BACKOFF = 20


def split_ids(ids: list[int], comma: int, chunk: int = CHUNK) -> list[list[int]]:
    """토큰 id를 chunk개 이하 조각으로 나눈다.

    자를 자리 앞 BACKOFF 토큰 안에 쉼표가 있으면 그 뒤에서 자른다.
    """
    parts: list[list[int]] = []
    start = 0
    while len(ids) - start > chunk:
        end = start + chunk
        for i in range(end - 1, max(start, end - BACKOFF) - 1, -1):
            if ids[i] == comma:
                end = i + 1
                break
        parts.append(ids[start:end])
        start = end
    parts.append(ids[start:])
    return parts


def frame(part: list[int], bos: int, eos: int, pad: int, chunk: int = CHUNK) -> list[int]:
    """조각에 시작·끝 토큰을 붙이고 chunk + 2 길이로 채운다."""
    return [bos, *part, eos] + [pad] * (chunk - len(part))


def encode(pipe: Any, prompt: str) -> tuple[Any, Any] | None:
    """긴 프롬프트의 (prompt_embeds, pooled_prompt_embeds).

    75토큰 이하면 None을 돌려줘 파이프라인 기본 인코딩을 쓴다.
    """
    import torch

    tok = pipe.tokenizer
    ids: list[int] = tok(prompt, truncation=False, add_special_tokens=False).input_ids
    if len(ids) <= CHUNK:
        return None
    comma = tok.convert_tokens_to_ids(",</w>")
    parts = split_ids(ids, comma)
    device = pipe._execution_device
    hidden = []
    pooled = None
    with torch.no_grad():
        for tokenizer, encoder in (
            (pipe.tokenizer, pipe.text_encoder),
            (pipe.tokenizer_2, pipe.text_encoder_2),
        ):
            pad = (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            )
            states = []
            for index, part in enumerate(parts):
                framed = frame(part, tokenizer.bos_token_id, tokenizer.eos_token_id, pad)
                out = encoder(torch.tensor([framed], device=device), output_hidden_states=True)
                states.append(out.hidden_states[-2])
                if encoder is pipe.text_encoder_2 and index == 0:
                    # SDXL pooled 임베딩은 두 번째 인코더의 첫 조각 투영값 (기본 인코딩과 같다)
                    pooled = out[0]
            hidden.append(torch.cat(states, dim=1))
    embeds = torch.cat(hidden, dim=-1).to(dtype=pipe.text_encoder_2.dtype)
    return embeds, pooled
