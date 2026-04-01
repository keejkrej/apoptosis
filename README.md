# apoptosis

`uv` workspace for apoptosis analysis, split into:

- `stain/`: stain-channel ROI fluorescence trace extraction
- `bf/`: brightfield morphology analysis placeholder

The current implemented workflow lives in `stain/` and reads an ND2 file directly,
applies `roi,x,y,w,h` bounding boxes for one position, and writes one long-form CSV
row per ROI per frame with raw sum, quantiles, `corrected`, and ND2-derived
timestamps.

Example:

```powershell
uv run --package apoptosis-stain apoptosis-roi-timeseries `
  "Z:\projects\LISCA\Experiments\20260327Apoptose\20260327_10x_Ti2_A549_STS.nd2" `
  "C:\Users\ctyja\data\20260327Apoptose\bbox\Pos0.csv" `
  --pos 0 `
  --channel 1
```
