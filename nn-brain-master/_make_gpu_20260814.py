"""RNN_tutorial_fixed_20260814.ipynb를 GPU(CUDA)에서도 돌아가게 수정한 사본 생성.

원본은 device 처리가 없어 항상 CPU에서 돈다. 아래 패치는 torch.cuda.is_available()
분기라 GPU가 없는 환경에서도 그대로 동작한다.

주의: 이 튜토리얼 크기(hidden=128, batch=16, seq=100)에서는 GPU가 오히려 느리다.
CTRNN이 시간 스텝 100번을 파이썬 for 루프로 돌기 때문에 스텝당 연산량(16x128 행렬곱)이
커널 실행 오버헤드보다 작다. hidden_size / batch_size를 키우면 역전된다.
"""

import json
from pathlib import Path

SRC = Path("RNN_tutorial_fixed_20260814.ipynb")
DST = Path("RNN_tutorial_gpu_20260814.ipynb")

# ---------------------------------------------------------------------------
# (셀 인덱스들, 원본 조각, 대체 조각) — 각 셀에서 정확히 1회 치환되어야 함
# ---------------------------------------------------------------------------
PATCHES = [
    # --- set_device(): mps 전용이라 CUDA를 못 잡는다. cuda -> mps -> cpu 순으로 ---
    ((11,),
     '  device = "mps" if torch.backends.mps.is_available() else "cpu"\n'
     '  if device != "mps":\n',
     '  if torch.cuda.is_available():\n'
     '    device = "cuda"                       # NVIDIA GPU\n'
     '  elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():\n'
     '    device = "mps"                        # Apple Silicon\n'
     '  else:\n'
     '    device = "cpu"\n'
     '  if device == "cpu":\n'),

    # --- 모델 생성부: 파라미터를 GPU로. 반드시 optimizer 생성 전에 해야 한다 ---
    ((31, 41),
     "net = RNNNet(input_size=input_size, hidden_size=hidden_size,\n"
     "             output_size=output_size, dt=env.dt)",
     "# .to(device): 파라미터를 GPU 메모리로 옮긴다. 반드시 optimizer 생성 전에.\n"
     "net = RNNNet(input_size=input_size, hidden_size=hidden_size,\n"
     "             output_size=output_size, dt=env.dt).to(device)"),
    ((51, 53, 58),
     "net = RNNNet(input_size=input_size, hidden_size=128, output_size=output_size, dt=env.dt)",
     "net = RNNNet(input_size=input_size, hidden_size=128, output_size=output_size, dt=env.dt).to(device)"),
    ((63,), "    net = maker()", "    net = maker().to(device)"),
    ((65,), "    net  = maker()", "    net  = maker().to(device)"),

    # --- 학습 루프의 배치도 모델과 같은 device에 있어야 한다 ---
    ((31, 41, 50),
     "        inputs = torch.from_numpy(inputs).type(torch.float)\n"
     "        labels = torch.from_numpy(labels.flatten()).type(torch.long)",
     "        # 입력/정답도 모델과 같은 device에 있어야 한다\n"
     "        inputs = torch.from_numpy(inputs).type(torch.float).to(device)\n"
     "        labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)"),
    ((63, 65),
     "        x = torch.from_numpy(x).type(torch.float)\n"
     "        y = torch.from_numpy(y.flatten()).type(torch.long)",
     "        x = torch.from_numpy(x).type(torch.float).to(device)\n"
     "        y = torch.from_numpy(y.flatten()).type(torch.long).to(device)"),

    # --- 분석/평가 루프의 입력 ---
    ((34, 42),
     "    inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)",
     "    inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)"),
    ((49,),
     "            inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)",
     "            inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)"),
    ((65,),
     "            pred, _ = net(torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float))",
     "            pred, _ = net(torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device))"),
    ((64,),
     "    _, (o1, o2, o3) = last_hier(torch.from_numpy(ev.ob[:, np.newaxis, :]).type(torch.float))",
     "    _, (o1, o2, o3) = last_hier(torch.from_numpy(ev.ob[:, np.newaxis, :]).type(torch.float).to(device))"),
    ((65,),
     "    _, mods = hier_net(torch.from_numpy(ev.ob[:, np.newaxis, :]).type(torch.float))",
     "    _, mods = hier_net(torch.from_numpy(ev.ob[:, np.newaxis, :]).type(torch.float).to(device))"),

    # --- GPU 텐서는 .cpu()를 거쳐야 numpy로 변환된다 ---
    ((34, 42),
     "    action_pred = action_pred.detach().numpy()[:, 0, :]",
     "    # GPU 텐서는 .numpy()를 바로 못 부른다. .cpu()로 먼저 내려야 한다.\n"
     "    action_pred = action_pred.detach().cpu().numpy()[:, 0, :]"),
    ((34, 42),
     "    rnn_activity = rnn_activity[:, 0, :].detach().numpy()",
     "    rnn_activity = rnn_activity[:, 0, :].detach().cpu().numpy()"),
    ((64, 65),
     "    a = o[:, 0, :].numpy()",
     "    a = o[:, 0, :].cpu().numpy()"),
]

# 셀 전체를 덧붙이는 패치: (셀 인덱스, 뒤에 붙일 코드)
APPENDS = [
    # 공통 import 셀 — 이후 대부분의 셀이 이 device 변수를 참조한다
    (9, "\n# GPU가 있으면 GPU, 없으면 CPU로 자동 선택\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "print('device:', device)\n"),
    # cell 65는 앞 셀에 의존하지 않는 자족 셀이라 device를 따로 정의한다
    (65, "\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
         "print('device:', device)\n"),
]

# ---------------------------------------------------------------------------
nb = json.loads(SRC.read_text(encoding="utf-8"))
cells = nb["cells"]
n_applied = 0

for idxs, old, new in PATCHES:
    for idx in idxs:
        src = "".join(cells[idx]["source"])
        if src.count(old) != 1:
            raise SystemExit(f"cell {idx}: 패턴이 {src.count(old)}회 발견됨 (1회여야 함)\n{old!r}")
        cells[idx]["source"] = src.replace(old, new).splitlines(keepends=True)
        n_applied += 1

for idx, extra in APPENDS:
    src = "".join(cells[idx]["source"]).rstrip("\n")
    cells[idx]["source"] = (src + "\n" + extra).splitlines(keepends=True)

# cell 65는 자족 셀이므로 device 정의가 import 바로 뒤(설정부 앞)에 와야 한다
src65 = "".join(cells[65]["source"])
anchor = 'import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)\n'
tail = ("\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "print('device:', device)\n")
assert src65.count(anchor) == 1 and src65.endswith(tail)
src65 = src65[: -len(tail)].rstrip("\n") + "\n"
src65 = src65.replace(anchor, anchor + tail.lstrip("\n"))
cells[65]["source"] = src65.splitlines(keepends=True)

# 기존 실행 결과 제거 (CPU에서 낸 결과라 그대로 두면 오해를 부른다)
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

# 개행은 LF로 고정 (원본과 동일)
DST.write_bytes((json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
print(f"생성: {DST}  (치환 {n_applied}건 + device 정의 {len(APPENDS)}곳)")
