# Beta Impact Analysis

## Summary
- Blocks `11` / `pca`: best beta = `100`, best quantized PPL = `24.7780`, delta vs baseline = `1.2774`, gain vs SQ = `2.4896`.
- Blocks `11` / `random`: best beta = `1`, best quantized PPL = `24.6978`, delta vs baseline = `1.1972`, gain vs SQ = `2.5698`.
- Blocks `8-9-10-11` / `pca`: best beta = `50`, best quantized PPL = `40.9251`, delta vs baseline = `17.4245`, gain vs SQ = `157.0132`.
- Blocks `8-9-10-11` / `random`: best beta = `1`, best quantized PPL = `34.0347`, delta vs baseline = `10.5341`, gain vs SQ = `163.9036`.

## Trend Notes
- Blocks `11` / `pca`: as beta moves from `0` to `100`, linear error drops from `0.1643` to `0.1047`; however, the best PPL occurs at beta = `100` and the worst at beta = `2`, so lower reconstruction error does not always translate into lower PPL.
- Blocks `11` / `random`: as beta moves from `0` to `100`, linear error drops from `0.1458` to `0.1169`; however, the best PPL occurs at beta = `1` and the worst at beta = `0`, so lower reconstruction error does not always translate into lower PPL.
- Blocks `8-9-10-11` / `pca`: as beta moves from `0` to `100`, linear error drops from `0.1531` to `0.1029`; however, the best PPL occurs at beta = `50` and the worst at beta = `0`, so lower reconstruction error does not always translate into lower PPL.
- Blocks `8-9-10-11` / `random`: as beta moves from `0` to `100`, linear error drops from `0.1226` to `0.1021`; however, the best PPL occurs at beta = `1` and the worst at beta = `50`, so lower reconstruction error does not always translate into lower PPL.
