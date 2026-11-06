---
name: precise-state-explanations
description: Explain time-indexed algorithms and stateful pipelines with exact event, frame, and state labels. Use for tracking, estimation, streaming, and similar technical explanations.
---

# Precise State Explanations

Before answering, map each claim to the event and state it describes.

- Name the anchor and target explicitly, such as frame 0 at `t0` and the next image at `t1`. Do not rely on “first frame” when it could mean either one.
- Distinguish initialization, prediction, measurement, optimization, and publication. State which operation produced each pose, velocity, bias, or other value.
- Do not attribute a prediction for `t1` to the anchor at `t0`. For example, an identity pose at first contact can coexist with an IMU-predicted pose for the following image.
- Check the event ordering and active configuration in available source or runtime evidence before describing executed behavior. If evidence is incomplete, state the uncertainty instead of using a convenient but unsupported label.
- Correct a discovered mislabel directly and use the precise label thereafter.
