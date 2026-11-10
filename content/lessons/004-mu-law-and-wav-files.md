---
title: Companding and audio files
slug: 004-mu-law-and-wav-files
part: "Part 1 — Sound and signals"
summary: Spend 8-bit levels where speech lives with mu-law companding, then turn samples into the bytes of a WAV file and back.
minutes: 40
gpu: false
objectives:
  - Explain why a uniform quantizer wastes levels on quiet speech, and how a log-like curve fixes it.
  - Encode a signal into the 256 mu-law codes by hand and in numpy, and decode it back.
  - Convert floats to 16-bit PCM without wrapping, and say why the scale is 32768.
  - Write and read a 16-bit WAV file with struct, including chunks you skip.
prereqs:
  - 003-quantization-and-dither
exercise: 004-mu-law-and-wav-files
---

# Companding and audio files

A telephone line carries speech at 8,000 samples per second, and each sample gets 8 bits. That is 64 kilobits per second. But [Quantization and dither](/lessons/003-quantization-and-dither) showed that a sine 40 dB below full scale loses 40 dB of signal-to-noise ratio, and speech spends most of its time near zero. Measured on an 8-bit linear quantizer, that quiet sine comes out at 11 dB, which is harsh. How does a phone make quiet speech sound clean in the same 8 bits?

The second question comes after the first. Once every sample is a number in a small range, how do those numbers become a file you can double-click? The bytes of a WAV file are not a mystery, and if you have built one, you never have to trust a library blindly to read your audio. Both questions have the same shape: how a float becomes a few bytes on a wire or a disk.

## What you already know

- A quantizer rounds to a grid of $2^N$ levels, and the error is about $\Delta^2/12$ whatever the signal is.
- A quiet signal gets a lower SNR, because the noise stays the same size while the signal shrinks.
- A logarithm turns a huge range of values into a small one.
- `quantize(x, bits)`, from the previous chapter, is provided in the starter file.

## Spend the levels where the samples are

A uniform quantizer puts levels evenly from $-1$ to $1$. But speech samples are not spread evenly. Most are small, and only a few reach the extremes. So a uniform quantizer gives the same number of levels to the loud extremes, which are rarely used, as to the quiet middle, where nearly every sample lives.

The fix is to stretch the quiet part before the quantizer sees it. **Companding** compresses the signal with a log-like curve, rounds with a uniform quantizer, and expands the result afterward. Because the stretched quiet range spans more steps, each quiet sample is rounded with a smaller error. The loud samples lose a little precision, but a loud sample can afford it: an error of the same size is a small fraction of a big value.

The **mu-law** curve, with $\mu = 255$, is the standard choice. The ITU-T G.711 telephone standard encodes a piecewise-linear approximation of it:

$$
F(x) = \operatorname{sgn}(x) \frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)}
$$

Read it aloud: keep the sign of $x$, and replace its size with the log of one plus $\mu$ times the size, divided by the log of one plus $\mu$ so that 1 maps to 1. The one inside the log keeps $F(0) = 0$. Figure 1 draws it. Near zero the curve is steep, which is the stretch: an input of 0.01, one percent of full scale, maps to 0.2285.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="The mu-law curve with mu 255 against a straight line: the curve climbs steeply near zero, so an input of 0.01 maps to 0.2285">
<line x1="64" y1="248.0" x2="576" y2="248.0" class="stroke-grid"/>
<text x="58" y="252.0" class="fig-small" text-anchor="end">0</text>
<line x1="64" y1="202.0" x2="576" y2="202.0" class="stroke-grid"/>
<text x="58" y="206.0" class="fig-small" text-anchor="end">0.2</text>
<line x1="64" y1="156.0" x2="576" y2="156.0" class="stroke-grid"/>
<text x="58" y="160.0" class="fig-small" text-anchor="end">0.4</text>
<line x1="64" y1="110.0" x2="576" y2="110.0" class="stroke-grid"/>
<text x="58" y="114.0" class="fig-small" text-anchor="end">0.6</text>
<line x1="64" y1="64.0" x2="576" y2="64.0" class="stroke-grid"/>
<text x="58" y="68.0" class="fig-small" text-anchor="end">0.8</text>
<line x1="64" y1="18.0" x2="576" y2="18.0" class="stroke-grid"/>
<text x="58" y="22.0" class="fig-small" text-anchor="end">1</text>
<line x1="64.0" y1="248" x2="64.0" y2="252" class="stroke-muted"/>
<text x="64.0" y="265" class="fig-small" text-anchor="middle">0</text>
<line x1="166.4" y1="248" x2="166.4" y2="252" class="stroke-muted"/>
<text x="166.4" y="265" class="fig-small" text-anchor="middle">0.2</text>
<line x1="268.8" y1="248" x2="268.8" y2="252" class="stroke-muted"/>
<text x="268.8" y="265" class="fig-small" text-anchor="middle">0.4</text>
<line x1="371.2" y1="248" x2="371.2" y2="252" class="stroke-muted"/>
<text x="371.2" y="265" class="fig-small" text-anchor="middle">0.6</text>
<line x1="473.6" y1="248" x2="473.6" y2="252" class="stroke-muted"/>
<text x="473.6" y="265" class="fig-small" text-anchor="middle">0.8</text>
<line x1="576.0" y1="248" x2="576.0" y2="252" class="stroke-muted"/>
<text x="576.0" y="265" class="fig-small" text-anchor="middle">1</text>
<line x1="64" y1="248" x2="576" y2="248" class="stroke-muted"/>
<line x1="64" y1="18" x2="64" y2="248" class="stroke-muted"/>
<text x="320.0" y="292" class="fig-small" text-anchor="middle">input amplitude |x|</text>
<text x="14" y="133.0" class="fig-small" text-anchor="middle" transform="rotate(-90 14 133.0)">compressed value F(x)</text>
<polyline points="64.0,248.0 65.3,247.4 66.6,246.8 67.8,246.3 69.1,245.7 70.4,245.1 71.7,244.5 73.0,244.0 74.3,243.4 75.5,242.8 76.8,242.2 78.1,241.7 79.4,241.1 80.7,240.5 82.0,239.9 83.2,239.4 84.5,238.8 85.8,238.2 87.1,237.6 88.4,237.0 89.7,236.5 90.9,235.9 92.2,235.3 93.5,234.7 94.8,234.2 96.1,233.6 97.4,233.0 98.6,232.4 99.9,231.9 101.2,231.3 102.5,230.7 103.8,230.1 105.1,229.6 106.3,229.0 107.6,228.4 108.9,227.8 110.2,227.2 111.5,226.7 112.8,226.1 114.0,225.5 115.3,224.9 116.6,224.4 117.9,223.8 119.2,223.2 120.5,222.6 121.7,222.1 123.0,221.5 124.3,220.9 125.6,220.3 126.9,219.8 128.2,219.2 129.4,218.6 130.7,218.0 132.0,217.4 133.3,216.9 134.6,216.3 135.9,215.7 137.1,215.1 138.4,214.6 139.7,214.0 141.0,213.4 142.3,212.8 143.6,212.3 144.8,211.7 146.1,211.1 147.4,210.5 148.7,210.0 150.0,209.4 151.3,208.8 152.5,208.2 153.8,207.6 155.1,207.1 156.4,206.5 157.7,205.9 159.0,205.3 160.2,204.8 161.5,204.2 162.8,203.6 164.1,203.0 165.4,202.5 166.7,201.9 167.9,201.3 169.2,200.7 170.5,200.2 171.8,199.6 173.1,199.0 174.4,198.4 175.6,197.8 176.9,197.3 178.2,196.7 179.5,196.1 180.8,195.5 182.1,195.0 183.3,194.4 184.6,193.8 185.9,193.2 187.2,192.7 188.5,192.1 189.8,191.5 191.0,190.9 192.3,190.4 193.6,189.8 194.9,189.2 196.2,188.6 197.5,188.1 198.7,187.5 200.0,186.9 201.3,186.3 202.6,185.7 203.9,185.2 205.2,184.6 206.4,184.0 207.7,183.4 209.0,182.9 210.3,182.3 211.6,181.7 212.9,181.1 214.1,180.6 215.4,180.0 216.7,179.4 218.0,178.8 219.3,178.3 220.6,177.7 221.8,177.1 223.1,176.5 224.4,175.9 225.7,175.4 227.0,174.8 228.3,174.2 229.5,173.6 230.8,173.1 232.1,172.5 233.4,171.9 234.7,171.3 235.9,170.8 237.2,170.2 238.5,169.6 239.8,169.0 241.1,168.5 242.4,167.9 243.6,167.3 244.9,166.7 246.2,166.1 247.5,165.6 248.8,165.0 250.1,164.4 251.3,163.8 252.6,163.3 253.9,162.7 255.2,162.1 256.5,161.5 257.8,161.0 259.0,160.4 260.3,159.8 261.6,159.2 262.9,158.7 264.2,158.1 265.5,157.5 266.7,156.9 268.0,156.3 269.3,155.8 270.6,155.2 271.9,154.6 273.2,154.0 274.4,153.5 275.7,152.9 277.0,152.3 278.3,151.7 279.6,151.2 280.9,150.6 282.1,150.0 283.4,149.4 284.7,148.9 286.0,148.3 287.3,147.7 288.6,147.1 289.8,146.5 291.1,146.0 292.4,145.4 293.7,144.8 295.0,144.2 296.3,143.7 297.5,143.1 298.8,142.5 300.1,141.9 301.4,141.4 302.7,140.8 304.0,140.2 305.2,139.6 306.5,139.1 307.8,138.5 309.1,137.9 310.4,137.3 311.7,136.7 312.9,136.2 314.2,135.6 315.5,135.0 316.8,134.4 318.1,133.9 319.4,133.3 320.6,132.7 321.9,132.1 323.2,131.6 324.5,131.0 325.8,130.4 327.1,129.8 328.3,129.3 329.6,128.7 330.9,128.1 332.2,127.5 333.5,126.9 334.8,126.4 336.0,125.8 337.3,125.2 338.6,124.6 339.9,124.1 341.2,123.5 342.5,122.9 343.7,122.3 345.0,121.8 346.3,121.2 347.6,120.6 348.9,120.0 350.2,119.5 351.4,118.9 352.7,118.3 354.0,117.7 355.3,117.1 356.6,116.6 357.9,116.0 359.1,115.4 360.4,114.8 361.7,114.3 363.0,113.7 364.3,113.1 365.6,112.5 366.8,112.0 368.1,111.4 369.4,110.8 370.7,110.2 372.0,109.7 373.3,109.1 374.5,108.5 375.8,107.9 377.1,107.3 378.4,106.8 379.7,106.2 381.0,105.6 382.2,105.0 383.5,104.5 384.8,103.9 386.1,103.3 387.4,102.7 388.7,102.2 389.9,101.6 391.2,101.0 392.5,100.4 393.8,99.9 395.1,99.3 396.4,98.7 397.6,98.1 398.9,97.5 400.2,97.0 401.5,96.4 402.8,95.8 404.1,95.2 405.3,94.7 406.6,94.1 407.9,93.5 409.2,92.9 410.5,92.4 411.7,91.8 413.0,91.2 414.3,90.6 415.6,90.1 416.9,89.5 418.2,88.9 419.4,88.3 420.7,87.7 422.0,87.2 423.3,86.6 424.6,86.0 425.9,85.4 427.1,84.9 428.4,84.3 429.7,83.7 431.0,83.1 432.3,82.6 433.6,82.0 434.8,81.4 436.1,80.8 437.4,80.3 438.7,79.7 440.0,79.1 441.3,78.5 442.5,77.9 443.8,77.4 445.1,76.8 446.4,76.2 447.7,75.6 449.0,75.1 450.2,74.5 451.5,73.9 452.8,73.3 454.1,72.8 455.4,72.2 456.7,71.6 457.9,71.0 459.2,70.5 460.5,69.9 461.8,69.3 463.1,68.7 464.4,68.2 465.6,67.6 466.9,67.0 468.2,66.4 469.5,65.8 470.8,65.3 472.1,64.7 473.3,64.1 474.6,63.5 475.9,63.0 477.2,62.4 478.5,61.8 479.8,61.2 481.0,60.7 482.3,60.1 483.6,59.5 484.9,58.9 486.2,58.4 487.5,57.8 488.7,57.2 490.0,56.6 491.3,56.0 492.6,55.5 493.9,54.9 495.2,54.3 496.4,53.7 497.7,53.2 499.0,52.6 500.3,52.0 501.6,51.4 502.9,50.9 504.1,50.3 505.4,49.7 506.7,49.1 508.0,48.6 509.3,48.0 510.6,47.4 511.8,46.8 513.1,46.2 514.4,45.7 515.7,45.1 517.0,44.5 518.3,43.9 519.5,43.4 520.8,42.8 522.1,42.2 523.4,41.6 524.7,41.1 526.0,40.5 527.2,39.9 528.5,39.3 529.8,38.8 531.1,38.2 532.4,37.6 533.7,37.0 534.9,36.4 536.2,35.9 537.5,35.3 538.8,34.7 540.1,34.1 541.4,33.6 542.6,33.0 543.9,32.4 545.2,31.8 546.5,31.3 547.8,30.7 549.1,30.1 550.3,29.5 551.6,29.0 552.9,28.4 554.2,27.8 555.5,27.2 556.8,26.6 558.0,26.1 559.3,25.5 560.6,24.9 561.9,24.3 563.2,23.8 564.5,23.2 565.7,22.6 567.0,22.0 568.3,21.5 569.6,20.9 570.9,20.3 572.2,19.7 573.4,19.2 574.7,18.6 576.0,18.0" class="stroke-muted dashed" fill="none"/>
<polyline points="64.0,248.0 65.3,227.5 66.6,213.8 67.8,203.6 69.1,195.4 70.4,188.5 71.7,182.6 73.0,177.5 74.3,172.9 75.5,168.8 76.8,165.0 78.1,161.6 79.4,158.4 80.7,155.5 82.0,152.7 83.2,150.1 84.5,147.7 85.8,145.4 87.1,143.2 88.4,141.2 89.7,139.2 90.9,137.3 92.2,135.5 93.5,133.8 94.8,132.1 96.1,130.5 97.4,129.0 98.6,127.5 99.9,126.1 101.2,124.7 102.5,123.4 103.8,122.1 105.1,120.8 106.3,119.6 107.6,118.4 108.9,117.3 110.2,116.2 111.5,115.1 112.8,114.0 114.0,113.0 115.3,112.0 116.6,111.0 117.9,110.0 119.2,109.1 120.5,108.2 121.7,107.3 123.0,106.4 124.3,105.5 125.6,104.7 126.9,103.8 128.2,103.0 129.4,102.2 130.7,101.5 132.0,100.7 133.3,99.9 134.6,99.2 135.9,98.5 137.1,97.8 138.4,97.0 139.7,96.4 141.0,95.7 142.3,95.0 143.6,94.4 144.8,93.7 146.1,93.1 147.4,92.4 148.7,91.8 150.0,91.2 151.3,90.6 152.5,90.0 153.8,89.4 155.1,88.9 156.4,88.3 157.7,87.7 159.0,87.2 160.2,86.6 161.5,86.1 162.8,85.6 164.1,85.0 165.4,84.5 166.7,84.0 167.9,83.5 169.2,83.0 170.5,82.5 171.8,82.0 173.1,81.5 174.4,81.1 175.6,80.6 176.9,80.1 178.2,79.7 179.5,79.2 180.8,78.8 182.1,78.3 183.3,77.9 184.6,77.4 185.9,77.0 187.2,76.6 188.5,76.2 189.8,75.7 191.0,75.3 192.3,74.9 193.6,74.5 194.9,74.1 196.2,73.7 197.5,73.3 198.7,72.9 200.0,72.5 201.3,72.2 202.6,71.8 203.9,71.4 205.2,71.0 206.4,70.6 207.7,70.3 209.0,69.9 210.3,69.6 211.6,69.2 212.9,68.8 214.1,68.5 215.4,68.1 216.7,67.8 218.0,67.5 219.3,67.1 220.6,66.8 221.8,66.4 223.1,66.1 224.4,65.8 225.7,65.5 227.0,65.1 228.3,64.8 229.5,64.5 230.8,64.2 232.1,63.9 233.4,63.6 234.7,63.2 235.9,62.9 237.2,62.6 238.5,62.3 239.8,62.0 241.1,61.7 242.4,61.4 243.6,61.1 244.9,60.8 246.2,60.6 247.5,60.3 248.8,60.0 250.1,59.7 251.3,59.4 252.6,59.1 253.9,58.9 255.2,58.6 256.5,58.3 257.8,58.0 259.0,57.8 260.3,57.5 261.6,57.2 262.9,57.0 264.2,56.7 265.5,56.4 266.7,56.2 268.0,55.9 269.3,55.7 270.6,55.4 271.9,55.1 273.2,54.9 274.4,54.6 275.7,54.4 277.0,54.1 278.3,53.9 279.6,53.7 280.9,53.4 282.1,53.2 283.4,52.9 284.7,52.7 286.0,52.5 287.3,52.2 288.6,52.0 289.8,51.7 291.1,51.5 292.4,51.3 293.7,51.0 295.0,50.8 296.3,50.6 297.5,50.4 298.8,50.1 300.1,49.9 301.4,49.7 302.7,49.5 304.0,49.3 305.2,49.0 306.5,48.8 307.8,48.6 309.1,48.4 310.4,48.2 311.7,48.0 312.9,47.7 314.2,47.5 315.5,47.3 316.8,47.1 318.1,46.9 319.4,46.7 320.6,46.5 321.9,46.3 323.2,46.1 324.5,45.9 325.8,45.7 327.1,45.5 328.3,45.3 329.6,45.1 330.9,44.9 332.2,44.7 333.5,44.5 334.8,44.3 336.0,44.1 337.3,43.9 338.6,43.7 339.9,43.5 341.2,43.3 342.5,43.1 343.7,42.9 345.0,42.7 346.3,42.6 347.6,42.4 348.9,42.2 350.2,42.0 351.4,41.8 352.7,41.6 354.0,41.5 355.3,41.3 356.6,41.1 357.9,40.9 359.1,40.7 360.4,40.6 361.7,40.4 363.0,40.2 364.3,40.0 365.6,39.8 366.8,39.7 368.1,39.5 369.4,39.3 370.7,39.1 372.0,39.0 373.3,38.8 374.5,38.6 375.8,38.5 377.1,38.3 378.4,38.1 379.7,38.0 381.0,37.8 382.2,37.6 383.5,37.5 384.8,37.3 386.1,37.1 387.4,37.0 388.7,36.8 389.9,36.6 391.2,36.5 392.5,36.3 393.8,36.2 395.1,36.0 396.4,35.8 397.6,35.7 398.9,35.5 400.2,35.4 401.5,35.2 402.8,35.0 404.1,34.9 405.3,34.7 406.6,34.6 407.9,34.4 409.2,34.3 410.5,34.1 411.7,34.0 413.0,33.8 414.3,33.7 415.6,33.5 416.9,33.4 418.2,33.2 419.4,33.1 420.7,32.9 422.0,32.8 423.3,32.6 424.6,32.5 425.9,32.3 427.1,32.2 428.4,32.0 429.7,31.9 431.0,31.7 432.3,31.6 433.6,31.5 434.8,31.3 436.1,31.2 437.4,31.0 438.7,30.9 440.0,30.7 441.3,30.6 442.5,30.5 443.8,30.3 445.1,30.2 446.4,30.1 447.7,29.9 449.0,29.8 450.2,29.6 451.5,29.5 452.8,29.4 454.1,29.2 455.4,29.1 456.7,29.0 457.9,28.8 459.2,28.7 460.5,28.6 461.8,28.4 463.1,28.3 464.4,28.2 465.6,28.0 466.9,27.9 468.2,27.8 469.5,27.6 470.8,27.5 472.1,27.4 473.3,27.2 474.6,27.1 475.9,27.0 477.2,26.9 478.5,26.7 479.8,26.6 481.0,26.5 482.3,26.3 483.6,26.2 484.9,26.1 486.2,26.0 487.5,25.8 488.7,25.7 490.0,25.6 491.3,25.5 492.6,25.3 493.9,25.2 495.2,25.1 496.4,25.0 497.7,24.9 499.0,24.7 500.3,24.6 501.6,24.5 502.9,24.4 504.1,24.2 505.4,24.1 506.7,24.0 508.0,23.9 509.3,23.8 510.6,23.6 511.8,23.5 513.1,23.4 514.4,23.3 515.7,23.2 517.0,23.1 518.3,22.9 519.5,22.8 520.8,22.7 522.1,22.6 523.4,22.5 524.7,22.4 526.0,22.2 527.2,22.1 528.5,22.0 529.8,21.9 531.1,21.8 532.4,21.7 533.7,21.6 534.9,21.5 536.2,21.3 537.5,21.2 538.8,21.1 540.1,21.0 541.4,20.9 542.6,20.8 543.9,20.7 545.2,20.6 546.5,20.5 547.8,20.3 549.1,20.2 550.3,20.1 551.6,20.0 552.9,19.9 554.2,19.8 555.5,19.7 556.8,19.6 558.0,19.5 559.3,19.4 560.6,19.3 561.9,19.2 563.2,19.0 564.5,18.9 565.7,18.8 567.0,18.7 568.3,18.6 569.6,18.5 570.9,18.4 572.2,18.3 573.4,18.2 574.7,18.1 576.0,18.0" class="stroke-b" fill="none" stroke-width="2"/>
<circle cx="69.1" cy="195.5" r="4" class="fill-a"/>
<circle cx="115.2" cy="112.1" r="4" class="fill-a"/>
<circle cx="320.0" cy="46.6" r="4" class="fill-a"/>
<text x="79.1" y="209.4" class="fig-small" text-anchor="start">0.01 → 0.2285</text>
<text x="125.2" y="126.1" class="fig-small" text-anchor="start">0.1 → 0.591</text>
<text x="330.0" y="62.6" class="fig-small" text-anchor="start">0.5 → 0.8757</text>
<text x="371.2" y="190.5" class="fig-small" text-anchor="start">dashed: no companding</text>
</svg>
Figure 1. The mu-law curve with mu equal to 255, against no companding (dashed). An input of 0.01 maps to 0.2285, and 0.1 maps to 0.591, so the quietest tenth of the input uses more than half of the output range.
```

To get integer codes, map $F(x)$ from $[-1, 1]$ to $[0, \mu]$ and round. This is how `torchaudio.functional.mu_law_encode` does it, and the code is the same two lines you will write:

```python
y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
codes = np.floor((y + 1) / 2 * mu + 0.5).astype(np.int64)
```

### Work one code by hand

Take $x = 0.01$. Then $\ln(1 + 2.55) = 1.2669$ and $\ln 256 = 5.5452$, so $F = 0.2285$. Map it to codes: $(0.2285 + 1)/2 \times 255 = 156.67$, plus 0.5 is 157.17, and the floor is 157. The following table shows five inputs and their codes.

| $x$ | $F(x)$ | Code |
|---|---|---|
| 0.001 | 0.0410 | 133 |
| 0.01 | 0.2285 | 157 |
| 0.1 | 0.5910 | 203 |
| 0.5 | 0.8757 | 239 |
| 1.0 | 1.0 | 255 |

Inputs between 0 and 0.01 get the codes 128 to 157, which is 12% of all 256 codes for one percent of the range. Decoding undoes the curve with a power, $F^{-1}(y) = \operatorname{sgn}(y)\,\big((1+\mu)^{|y|} - 1\big)/\mu$, after mapping the code back to $y = 2c/\mu - 1$.

> [!TIP]
> Code 128 decodes to about 0.000086, not to 0, because 256 codes have no middle one. Do not test for exact silence after a mu-law round trip.

Figure 2 measures the payoff on a sine at three levels. At full scale, linear 8-bit wins, 49.7 dB against 37.7 dB. But at $-20$ dB the two have swapped, 29.4 against 36.9. At $-40$ dB, mu-law keeps 34.0 dB while linear has fallen to 11.1. Mu-law gives up some quality on loud sounds to hold nearly constant quality as the level drops, which is what speech needs.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="SNR of an 8-bit sine: linear gives 49.7, 29.4, and 11.1 dB at 0, minus 20, and minus 40 dB; mu-law gives 37.7, 36.9, and 34.0 dB">
<text x="40" y="22" class="fig-bold" text-anchor="start">SNR of an 8-bit sine at three levels (dB)</text>
<rect x="380" y="10" width="12" height="12" class="fill-a"/>
<text x="398" y="21" class="fig-small" text-anchor="start">linear</text>
<rect x="470" y="10" width="12" height="12" class="fill-b"/>
<text x="488" y="21" class="fig-small" text-anchor="start">mu-law</text>
<text x="40" y="86" class="fig-small" text-anchor="start">0 dB (0.99)</text>
<rect x="130" y="52" width="361.7" height="24" class="fill-a"/>
<text x="499.7" y="69" class="fig-small" text-anchor="start">49.7</text>
<rect x="130" y="82" width="273.9" height="24" class="fill-b"/>
<text x="411.9" y="99" class="fig-small" text-anchor="start">37.7</text>
<text x="40" y="164" class="fig-small" text-anchor="start">−20 dB</text>
<rect x="130" y="130" width="214.0" height="24" class="fill-a"/>
<text x="352.0" y="147" class="fig-small" text-anchor="start">29.4</text>
<rect x="130" y="160" width="268.3" height="24" class="fill-b"/>
<text x="406.3" y="177" class="fig-small" text-anchor="start">36.9</text>
<text x="40" y="242" class="fig-small" text-anchor="start">−40 dB</text>
<rect x="130" y="208" width="80.8" height="24" class="fill-a"/>
<text x="218.8" y="225" class="fig-small" text-anchor="start">11.1</text>
<rect x="130" y="238" width="247.3" height="24" class="fill-b"/>
<text x="385.3" y="255" class="fig-small" text-anchor="start">34.0</text>
</svg>
Figure 2. Measured SNR of an 8-bit sine at three levels. Linear quantization loses 20 dB for every 20 dB of level. Mu-law stays within 4 dB of itself across the whole range.
```

WaveNet used exactly this. It predicts each sample with a softmax, and a softmax over 65,536 16-bit values was impractical. Its authors compressed with mu-law and quantized to 256 values, and reported much better quality than 256 linear levels gave. You build that model in [WaveNet](/lessons/046-wavenet).

> [!KEY]
> A uniform quantizer has the same error at every level, so quiet samples suffer. Mu-law stretches the quiet range before rounding, so the error grows with the sample and the SNR stays nearly constant as the level falls.

## Turn samples into bytes

Now the second question. The common convention converts a float to a **16-bit PCM** integer, where PCM stands for pulse-code modulation: multiply by 32768, round, and clip to $[-32768, 32767]$. To convert back, divide by 32768. Why 32768 and not 32767? Because the integer range is lopsided: it runs one further on the negative side. Multiplying by 32768 puts $-1.0$ exactly on $-32768$ and makes every integer round-trip. Whisper's audio loader divides by exactly 32768.0.

The clip matters. A float of 1.0 times 32768 is 32768, which does not fit in 16 bits, and the cast wraps it to $-32768$. A loud peak then jumps from the top to the bottom of the scale, which you hear as a click. Work three values: 0.5 gives 16384, $-1.0$ gives $-32768$, and 1.0 clips to 32767.

A **WAV** file stores these integers in a RIFF container: a sequence of **chunks**, each a four-letter id, a 4-byte little-endian size, and that many bytes. Little-endian means the low byte comes first, so 16,000 is stored as `80 3e 00 00`. Figure 3 shows the 56 bytes of a file that holds three stereo frames at 16 kHz.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="The 56 bytes of a WAV file with three stereo frames at 16 kilohertz, in hexadecimal and split into fields: the RIFF header, the fmt chunk, and the data chunk with 12 sample bytes">
<text x="14" y="22" class="fig-small fig-bold" text-anchor="start">RIFF header</text>
<rect x="14" y="30" width="85" height="32" class="fill-soft-a stroke-muted"/>
<text x="56.5" y="51" class="fig-mono fig-small" text-anchor="middle">52 49 46 46</text>
<text x="56.5" y="78" class="fig-small" text-anchor="middle">RIFF</text>
<rect x="102" y="30" width="85" height="32" class="fill-soft-b stroke-muted"/>
<text x="144.5" y="51" class="fig-mono fig-small" text-anchor="middle">30 00 00 00</text>
<text x="144.5" y="78" class="fig-small" text-anchor="middle">size = 48</text>
<rect x="190" y="30" width="85" height="32" class="fill-soft-a stroke-muted"/>
<text x="232.5" y="51" class="fig-mono fig-small" text-anchor="middle">57 41 56 45</text>
<text x="232.5" y="78" class="fig-small" text-anchor="middle">WAVE</text>
<text x="14" y="108" class="fig-small fig-bold" text-anchor="start">fmt chunk</text>
<rect x="14" y="116" width="85" height="32" class="fill-soft-a stroke-muted"/>
<text x="56.5" y="137" class="fig-mono fig-small" text-anchor="middle">66 6d 74 20</text>
<text x="56.5" y="164" class="fig-small" text-anchor="middle">fmt </text>
<rect x="102" y="116" width="85" height="32" class="fill-soft-b stroke-muted"/>
<text x="144.5" y="137" class="fig-mono fig-small" text-anchor="middle">10 00 00 00</text>
<text x="144.5" y="164" class="fig-small" text-anchor="middle">size = 16</text>
<rect x="190" y="116" width="41" height="32" class="fill-soft-c stroke-muted"/>
<text x="210.5" y="137" class="fig-mono fig-small" text-anchor="middle">01 00</text>
<text x="210.5" y="164" class="fig-small" text-anchor="middle">tag 1</text>
<rect x="234" y="116" width="41" height="32" class="fill-soft-c stroke-muted"/>
<text x="254.5" y="137" class="fig-mono fig-small" text-anchor="middle">02 00</text>
<text x="254.5" y="164" class="fig-small" text-anchor="middle">2 ch</text>
<rect x="278" y="116" width="85" height="32" class="fill-soft-c stroke-muted"/>
<text x="320.5" y="137" class="fig-mono fig-small" text-anchor="middle">80 3e 00 00</text>
<text x="320.5" y="164" class="fig-small" text-anchor="middle">16000 Hz</text>
<rect x="366" y="116" width="85" height="32" class="fill-soft-c stroke-muted"/>
<text x="408.5" y="137" class="fig-mono fig-small" text-anchor="middle">00 fa 00 00</text>
<text x="408.5" y="164" class="fig-small" text-anchor="middle">bytes/s</text>
<rect x="454" y="116" width="41" height="32" class="fill-soft-c stroke-muted"/>
<text x="474.5" y="137" class="fig-mono fig-small" text-anchor="middle">04 00</text>
<text x="474.5" y="164" class="fig-small" text-anchor="middle">align</text>
<rect x="498" y="116" width="41" height="32" class="fill-soft-c stroke-muted"/>
<text x="518.5" y="137" class="fig-mono fig-small" text-anchor="middle">10 00</text>
<text x="518.5" y="164" class="fig-small" text-anchor="middle">bits</text>
<text x="14" y="194" class="fig-small fig-bold" text-anchor="start">data chunk</text>
<rect x="14" y="202" width="85" height="32" class="fill-soft-a stroke-muted"/>
<text x="56.5" y="223" class="fig-mono fig-small" text-anchor="middle">64 61 74 61</text>
<text x="56.5" y="250" class="fig-small" text-anchor="middle">data</text>
<rect x="102" y="202" width="85" height="32" class="fill-soft-b stroke-muted"/>
<text x="144.5" y="223" class="fig-mono fig-small" text-anchor="middle">0c 00 00 00</text>
<text x="144.5" y="250" class="fig-small" text-anchor="middle">size = 12</text>
<rect x="190" y="202" width="261" height="32" class="fill-soft-d stroke-muted"/>
<text x="320.5" y="223" class="fig-mono fig-small" text-anchor="middle">01 00 ff ff 02 00 fe ff 03 00 fd ff</text>
<text x="320.5" y="250" class="fig-small" text-anchor="middle">L0 R0 L1 R1 L2 R2</text>
</svg>
Figure 3. A complete WAV file, byte by byte. The 12-byte RIFF header holds the size of everything after it, 48. The fmt chunk describes the samples, and the data chunk holds them interleaved: left 0, right 0, left 1, right 1, and so on.
```

Read it from the top. The file says `RIFF`, then how many bytes follow, then `WAVE`. The `fmt ` chunk is 16 bytes: a format tag of 1 for integer PCM, the channel count, the sample rate, the bytes per second (rate times channels times 2), the bytes per frame (channels times 2), and the bits per sample, 16. The `data` chunk then holds the samples. The size in bytes 4 to 7 is $36 + 12 = 48$, which counts everything after those four bytes.

The `struct` module packs the fields. `<` means little-endian, `H` is a uint16, and `I` is a uint32:

```python
fmt = struct.pack("<HHIIHH", 1, channels, rate, rate * 2 * channels, 2 * channels, 16)
header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt
```

A numpy array shaped `(frames, channels)` is already in interleaved order, so `pcm.astype("<i2").tobytes()` writes it directly.

> [!WARNING]
> Files often carry extra chunks, such as `LIST` for metadata, before the data. A reader that seeks to byte 44 treats that metadata as audio. Walk the chunks instead, and remember that a chunk with an odd size is followed by one padding byte.

Other formats trade size for fidelity. **FLAC** is lossless: it predicts each block with a linear predictor and stores the small residual. **MP3** and **Opus** are lossy, and their decoders add a delay at the start, so a decoded clip no longer lines up sample for sample with the original.

> [!KEY]
> Back to both questions. Mu-law gives quiet speech clean 8-bit codes by spending levels where the samples are. And a WAV file is not magic: it is a 44-byte header of little-endian fields followed by interleaved int16 samples, which you can write and parse yourself.

> [!RECAP]
> - Mu-law codes are `np.floor((F(x) + 1) / 2 * mu + 0.5)`, and decoding inverts both steps.
> - Mu-law holds 8-bit SNR nearly constant across levels, where linear 8-bit loses 20 dB per 20 dB.
> - 16-bit PCM is `np.clip(np.round(x * 32768), -32768, 32767)`, and the clip prevents a wrap to $-32768$.
> - A 16-bit WAV file is a 44-byte header of little-endian fields, then interleaved int16 samples.
> - Readers must walk chunks, and skip the padding byte after an odd-sized one.

## Build it step by step

The checks test the functions in this order. `quantize` is provided.

1. **`mu_law_encode(x, mu)`** clips, compands with `np.log1p`, and maps to int64 codes. Common mistake: `np.round` instead of `np.floor(... + 0.5)`, which rounds 0.5 to the even neighbor and moves a few codes.
2. **`mu_law_decode(codes, mu)`** maps codes to $[-1, 1]$ and expands. Common mistake: forgetting to convert the codes to float before dividing.
3. **`float_to_pcm16(x)`** and **`pcm16_to_float(pcm)`** scale by 32768. Clip before `astype(np.int16)`. Common mistake: casting without clipping, which wraps 1.0 to $-32768$.
4. **`wav_bytes(pcm, sample_rate)`** packs the header with `struct.pack` and appends `pcm.astype("<i2").tobytes()`. Common mistake: writing 44 plus the data length in bytes 4 to 7 instead of 36 plus it.
5. **`read_wav_bytes(data)`** starts at byte 12 and loops over chunks with `struct.unpack("<I", ...)`. Use `np.frombuffer(body, dtype="<i2")` and reshape to `(-1, channels)`. Common mistake: forgetting the padding byte after an odd-sized chunk.

## Where this breaks

**Readers disagree on scaling.** `scipy.io.wavfile.read` returns raw integers, and `soundfile.read` returns floats in $[-1, 1)$. A 24-bit file often reads as left-justified `int32`, so its scale is $2^{31}$, not $2^{23}$. Check the dtype of everything you load.

**Layouts disagree too.** `soundfile` returns `(frames, channels)`, while torchaudio and `speechlab.play` use `(channels, frames)`. Downmix to mono with the mean over channels, not the sum, or the result can clip.

**Mu-law is for speech.** It spends levels near zero, which suits speech and wastes them on a signal that is always loud, such as a heavily compressed song. Bit rate is the other cost: G.711 sends 8-bit mu-law at 8 kHz, which is 64 kilobits per second, and the neural codecs later in the course send speech at a few kilobits per second.

## The exercise

You build `mu_law_encode`, `mu_law_decode`, `float_to_pcm16`, `pcm16_to_float`, `wav_bytes`, and `read_wav_bytes`. Press **Run** to compare 8-bit linear and mu-law on a quiet plucked note, and to write, read, and play a stereo WAV file.

The checks test the mu-law codes in the table, the end codes of the decoder, a round-trip error under 4%, and a 10 dB win over linear on quiet samples. They compare your PCM scaling on every int16 value and check the header of the 56-byte file in Figure 3. The last two checks compare your file with the `wave` module's byte for byte, and your reader on a `soundfile` stereo file and on a file with a padded `LIST` chunk.
