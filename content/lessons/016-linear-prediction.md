---
title: Linear prediction
slug: 016-linear-prediction
part: "Part 2 — Hearing and speech production"
summary: Predict each sample of a vowel from the ones before it, then read the vocal tract's envelope and formants out of the predictor.
minutes: 45
gpu: false
objectives:
  - Explain why a vowel is predictable from its past samples, and why the prediction error is the glottal source.
  - Compute autocorrelation LPC with the Levinson-Durbin recursion, and check it by hand on a two-coefficient example.
  - Turn the coefficients into a smooth spectral envelope, and read formants from the angles and radii of the poles.
  - "Explain what the real cepstrum adds: the pitch period as a peak at quefrency fs / F0."
prereqs:
  - 009-convolution-and-filters
  - 013-mfccs
  - 015-the-source-filter-model
exercise: 016-linear-prediction
---

# Linear prediction

Look at the spectrum of a vowel in Figure 1. The blue curve has about thirty sharp peaks between 0 and 4 kHz, and only three of them matter. The vowel's identity lives in the smooth hump the peaks ride on, the shape of the vocal tract. The peaks themselves are the harmonics of the voice, and they move whenever the speaker changes pitch.

So how do you read the formants without being fooled by the harmonics? Picking the three tallest peaks does not work, because they are single harmonics. You need a method that fits the *envelope*, the orange curve in the figure, and ignores the ripple.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Spectrum of a vowel frame with two LPC envelopes of order 18 and order 4, and the three true formant frequencies">
<line x1="60" y1="252.8" x2="570" y2="252.8" class="stroke-grid"/>
<text x="54" y="256.8" class="fig-small" text-anchor="end">-60</text>
<line x1="60" y1="190.2" x2="570" y2="190.2" class="stroke-grid"/>
<text x="54" y="194.2" class="fig-small" text-anchor="end">-40</text>
<line x1="60" y1="127.8" x2="570" y2="127.8" class="stroke-grid"/>
<text x="54" y="131.8" class="fig-small" text-anchor="end">-20</text>
<line x1="60" y1="65.2" x2="570" y2="65.2" class="stroke-grid"/>
<text x="54" y="69.2" class="fig-small" text-anchor="end">0</text>
<line x1="60" y1="284" x2="570" y2="284" class="stroke-muted"/>
<line x1="60" y1="34" x2="60" y2="284" class="stroke-muted"/>
<line x1="60.0" y1="284" x2="60.0" y2="288" class="stroke-muted"/>
<text x="60.0" y="301" class="fig-small" text-anchor="middle">0</text>
<line x1="123.8" y1="284" x2="123.8" y2="288" class="stroke-muted"/>
<text x="123.8" y="301" class="fig-small" text-anchor="middle">500</text>
<line x1="187.5" y1="284" x2="187.5" y2="288" class="stroke-muted"/>
<text x="187.5" y="301" class="fig-small" text-anchor="middle">1000</text>
<line x1="251.2" y1="284" x2="251.2" y2="288" class="stroke-muted"/>
<text x="251.2" y="301" class="fig-small" text-anchor="middle">1500</text>
<line x1="315.0" y1="284" x2="315.0" y2="288" class="stroke-muted"/>
<text x="315.0" y="301" class="fig-small" text-anchor="middle">2000</text>
<line x1="378.8" y1="284" x2="378.8" y2="288" class="stroke-muted"/>
<text x="378.8" y="301" class="fig-small" text-anchor="middle">2500</text>
<line x1="442.5" y1="284" x2="442.5" y2="288" class="stroke-muted"/>
<text x="442.5" y="301" class="fig-small" text-anchor="middle">3000</text>
<line x1="506.2" y1="284" x2="506.2" y2="288" class="stroke-muted"/>
<text x="506.2" y="301" class="fig-small" text-anchor="middle">3500</text>
<line x1="570.0" y1="284" x2="570.0" y2="288" class="stroke-muted"/>
<text x="570.0" y="301" class="fig-small" text-anchor="middle">4000</text>
<text x="315.0" y="318" text-anchor="middle">frequency (Hz)</text>
<text x="60" y="26" class="fig-small" text-anchor="start">power (dB, 0 = strongest harmonic)</text>
<line x1="153.1" y1="34" x2="153.1" y2="284" class="stroke-muted dashed"/>
<line x1="199.0" y1="34" x2="199.0" y2="284" class="stroke-muted dashed"/>
<line x1="371.1" y1="34" x2="371.1" y2="284" class="stroke-muted dashed"/>
<polyline points="60.0,142.6 62.0,148.4 64.0,167.2 66.0,202.5 68.0,232.6 70.0,187.8 72.0,158.3 73.9,144.0 75.9,142.0 77.9,152.1 79.9,176.1 81.9,214.5 83.9,212.7 85.9,172.4 87.9,148.7 89.9,138.5 91.9,140.2 93.9,154.8 95.9,185.5 97.9,217.8 99.8,189.1 101.8,156.1 103.8,137.6 105.8,131.2 107.8,136.7 109.8,156.4 111.8,193.7 113.8,205.3 115.8,164.3 117.8,137.5 119.8,123.6 121.8,120.9 123.8,130.3 125.7,156.1 127.7,188.5 129.7,173.4 131.7,134.4 133.7,112.5 135.7,102.8 137.7,104.1 139.7,118.0 141.7,151.6 143.7,158.8 145.7,119.8 147.7,88.6 149.6,71.1 151.6,65.2 153.6,70.7 155.6,89.8 157.6,138.2 159.6,142.2 161.6,123.7 163.6,101.0 165.6,87.9 167.6,86.2 169.6,96.0 171.6,119.4 173.6,152.5 175.5,151.6 177.5,123.5 179.5,100.8 181.5,91.0 183.5,92.8 185.5,107.2 187.5,138.0 189.5,147.9 191.5,125.7 193.5,98.8 195.5,81.9 197.5,76.2 199.5,81.7 201.4,100.8 203.4,148.5 205.4,149.8 207.4,140.2 209.4,122.2 211.4,110.4 213.4,109.3 215.4,118.8 217.4,140.4 219.4,172.7 221.4,179.3 223.4,167.1 225.4,143.2 227.3,133.7 229.3,136.4 231.3,150.4 233.3,175.8 235.3,194.1 237.3,194.8 239.3,175.2 241.3,155.6 243.3,149.4 245.3,155.7 247.3,174.9 249.3,203.1 251.2,208.6 253.2,199.9 255.2,179.5 257.2,164.8 259.2,161.8 261.2,171.6 263.2,196.0 265.2,221.0 267.2,219.3 269.2,199.9 271.2,182.2 273.2,172.2 275.2,172.4 277.1,185.4 279.1,212.3 281.1,226.8 283.1,224.5 285.1,198.2 287.1,183.7 289.1,178.4 291.1,182.2 293.1,198.0 295.1,220.1 297.1,228.1 299.1,223.2 301.1,195.8 303.0,184.3 305.0,183.4 307.0,191.4 309.0,209.9 311.0,222.2 313.0,227.8 315.0,217.2 317.0,192.6 319.0,183.9 321.0,187.0 323.0,200.1 325.0,220.7 327.0,222.1 328.9,224.0 330.9,207.8 332.9,187.9 334.9,181.8 336.9,188.4 338.9,207.8 340.9,226.3 342.9,219.1 344.9,212.5 346.9,194.2 348.9,179.5 350.9,176.3 352.9,186.4 354.8,212.0 356.8,217.0 358.8,207.2 360.8,190.7 362.8,174.1 364.8,164.8 366.8,165.6 368.8,180.5 370.8,218.2 372.8,203.8 374.8,201.3 376.8,191.9 378.8,179.8 380.7,176.0 382.7,181.7 384.7,198.7 386.7,224.5 388.7,221.8 390.7,228.0 392.7,215.9 394.7,201.1 396.7,199.7 398.7,211.2 400.7,234.5 402.7,245.9 404.6,238.3 406.6,243.7 408.6,231.4 410.6,216.4 412.6,216.8 414.6,234.6 416.6,272.9 418.6,266.4 420.6,250.5 422.6,250.4 424.6,243.9 426.6,229.6 428.6,230.3 430.5,252.3 432.5,277.1 434.5,289.0 436.5,258.7 438.5,253.5 440.5,254.7 442.5,242.6 444.5,242.5 446.5,263.0 448.5,268.6 450.5,310.0 452.5,263.9 454.5,255.7 456.4,262.6 458.4,256.9 460.4,254.4 462.4,269.7 464.4,266.8 466.4,308.9 468.4,268.2 470.4,258.0 472.4,266.3 474.4,274.6 476.4,267.2 478.4,276.3 480.4,268.0 482.3,303.3 484.3,272.9 486.3,261.0 488.3,267.3 490.3,300.4 492.3,280.7 494.3,284.0 496.3,270.4 498.3,299.8 500.3,279.1 502.3,264.8 504.3,267.7 506.2,319.9 508.2,290.8 510.2,292.9 512.2,273.0 514.2,296.5 516.2,287.7 518.2,269.8 520.2,268.9 522.2,300.2 524.2,290.8 526.2,301.4 528.2,275.3 530.2,292.9 532.1,298.9 534.1,276.1 536.1,271.1 538.1,292.0 540.1,286.6 542.1,306.5 544.1,277.0 546.1,289.6 548.1,308.7 550.1,283.9 552.1,274.3 554.1,289.3 556.1,283.6 558.0,307.2 560.0,278.3 562.0,287.1 564.0,305.1 566.0,293.5 568.0,278.4 570.0,289.4" class="stroke-a" fill="none"/>
<polyline points="60.0,157.1 62.0,157.1 64.0,157.0 66.0,156.9 68.0,156.8 70.0,156.6 72.0,156.4 73.9,156.2 75.9,155.9 77.9,155.6 79.9,155.2 81.9,154.8 83.9,154.4 85.9,153.9 87.9,153.3 89.9,152.7 91.9,152.1 93.9,151.4 95.9,150.7 97.9,149.9 99.8,149.0 101.8,148.1 103.8,147.1 105.8,146.1 107.8,145.0 109.8,143.8 111.8,142.5 113.8,141.1 115.8,139.6 117.8,138.0 119.8,136.3 121.8,134.5 123.8,132.5 125.7,130.4 127.7,128.1 129.7,125.6 131.7,122.8 133.7,119.8 135.7,116.5 137.7,112.7 139.7,108.5 141.7,103.7 143.7,98.2 145.7,91.8 147.7,84.5 149.6,76.9 151.6,72.1 153.6,73.6 155.6,79.3 157.6,85.3 159.6,90.4 161.6,94.6 163.6,98.0 165.6,100.6 167.6,102.7 169.6,104.4 171.6,105.6 173.6,106.3 175.5,106.7 177.5,106.8 179.5,106.4 181.5,105.7 183.5,104.5 185.5,102.9 187.5,100.7 189.5,98.0 191.5,94.6 193.5,90.8 195.5,87.1 197.5,85.2 199.5,87.0 201.4,92.1 203.4,98.6 205.4,105.0 207.4,110.9 209.4,116.2 211.4,121.0 213.4,125.3 215.4,129.3 217.4,132.9 219.4,136.3 221.4,139.5 223.4,142.4 225.4,145.2 227.3,147.7 229.3,150.2 231.3,152.5 233.3,154.7 235.3,156.8 237.3,158.8 239.3,160.7 241.3,162.5 243.3,164.3 245.3,165.9 247.3,167.6 249.3,169.1 251.2,170.6 253.2,172.0 255.2,173.4 257.2,174.7 259.2,176.0 261.2,177.2 263.2,178.4 265.2,179.5 267.2,180.6 269.2,181.7 271.2,182.7 273.2,183.7 275.2,184.6 277.1,185.6 279.1,186.4 281.1,187.3 283.1,188.1 285.1,188.9 287.1,189.6 289.1,190.3 291.1,191.0 293.1,191.7 295.1,192.3 297.1,192.8 299.1,193.4 301.1,193.9 303.0,194.4 305.0,194.8 307.0,195.2 309.0,195.6 311.0,195.9 313.0,196.2 315.0,196.4 317.0,196.6 319.0,196.8 321.0,196.9 323.0,196.9 325.0,196.9 327.0,196.9 328.9,196.8 330.9,196.6 332.9,196.3 334.9,196.0 336.9,195.6 338.9,195.1 340.9,194.5 342.9,193.9 344.9,193.1 346.9,192.2 348.9,191.1 350.9,189.9 352.9,188.6 354.8,187.1 356.8,185.5 358.8,183.7 360.8,181.8 362.8,179.8 364.8,178.0 366.8,176.5 368.8,175.6 370.8,175.8 372.8,177.1 374.8,179.4 376.8,182.5 378.8,185.9 380.7,189.5 382.7,193.1 384.7,196.6 386.7,199.9 388.7,203.1 390.7,206.1 392.7,208.9 394.7,211.6 396.7,214.2 398.7,216.7 400.7,219.1 402.7,221.3 404.6,223.5 406.6,225.5 408.6,227.5 410.6,229.5 412.6,231.3 414.6,233.1 416.6,234.8 418.6,236.5 420.6,238.1 422.6,239.7 424.6,241.2 426.6,242.7 428.6,244.1 430.5,245.5 432.5,246.8 434.5,248.1 436.5,249.4 438.5,250.6 440.5,251.8 442.5,253.0 444.5,254.2 446.5,255.3 448.5,256.4 450.5,257.4 452.5,258.4 454.5,259.4 456.4,260.4 458.4,261.4 460.4,262.3 462.4,263.2 464.4,264.1 466.4,264.9 468.4,265.8 470.4,266.6 472.4,267.4 474.4,268.1 476.4,268.9 478.4,269.6 480.4,270.3 482.3,271.0 484.3,271.7 486.3,272.4 488.3,273.0 490.3,273.6 492.3,274.2 494.3,274.8 496.3,275.4 498.3,275.9 500.3,276.5 502.3,277.0 504.3,277.5 506.2,278.0 508.2,278.5 510.2,278.9 512.2,279.4 514.2,279.8 516.2,280.2 518.2,280.6 520.2,281.0 522.2,281.4 524.2,281.8 526.2,282.1 528.2,282.5 530.2,282.8 532.1,283.1 534.1,283.4 536.1,283.7 538.1,284.0 540.1,284.3 542.1,284.6 544.1,284.8 546.1,285.1 548.1,285.3 550.1,285.5 552.1,285.7 554.1,285.9 556.1,286.1 558.0,286.3 560.0,286.5 562.0,286.7 564.0,286.8 566.0,287.0 568.0,287.2 570.0,287.3" class="stroke-b" fill="none"/>
<polyline points="60.0,136.3 62.0,136.3 64.0,136.3 66.0,136.2 68.0,136.1 70.0,136.0 72.0,135.9 73.9,135.7 75.9,135.5 77.9,135.3 79.9,135.0 81.9,134.8 83.9,134.5 85.9,134.1 87.9,133.8 89.9,133.4 91.9,133.0 93.9,132.5 95.9,132.0 97.9,131.5 99.8,131.0 101.8,130.4 103.8,129.7 105.8,129.1 107.8,128.4 109.8,127.6 111.8,126.9 113.8,126.0 115.8,125.1 117.8,124.2 119.8,123.2 121.8,122.2 123.8,121.1 125.7,119.9 127.7,118.7 129.7,117.4 131.7,116.0 133.7,114.5 135.7,112.9 137.7,111.3 139.7,109.5 141.7,107.6 143.7,105.5 145.7,103.4 147.7,101.0 149.6,98.5 151.6,95.8 153.6,92.9 155.6,89.9 157.6,86.8 159.6,83.8 161.6,81.1 163.6,79.0 165.6,78.1 167.6,78.5 169.6,80.0 171.6,82.3 173.6,85.0 175.5,87.8 177.5,90.6 179.5,93.3 181.5,95.8 183.5,98.1 185.5,100.2 187.5,102.2 189.5,104.1 191.5,105.8 193.5,107.5 195.5,109.0 197.5,110.4 199.5,111.8 201.4,113.1 203.4,114.4 205.4,115.6 207.4,116.7 209.4,117.9 211.4,119.0 213.4,120.1 215.4,121.2 217.4,122.3 219.4,123.4 221.4,124.5 223.4,125.6 225.4,126.7 227.3,127.9 229.3,129.1 231.3,130.3 233.3,131.5 235.3,132.8 237.3,134.0 239.3,135.3 241.3,136.7 243.3,138.0 245.3,139.4 247.3,140.7 249.3,142.1 251.2,143.5 253.2,144.9 255.2,146.3 257.2,147.7 259.2,149.1 261.2,150.6 263.2,152.0 265.2,153.4 267.2,154.8 269.2,156.2 271.2,157.5 273.2,158.9 275.2,160.3 277.1,161.6 279.1,162.9 281.1,164.3 283.1,165.6 285.1,166.9 287.1,168.1 289.1,169.4 291.1,170.7 293.1,171.9 295.1,173.1 297.1,174.3 299.1,175.5 301.1,176.7 303.0,177.9 305.0,179.0 307.0,180.2 309.0,181.3 311.0,182.4 313.0,183.5 315.0,184.6 317.0,185.7 319.0,186.7 321.0,187.8 323.0,188.8 325.0,189.9 327.0,190.9 328.9,191.9 330.9,192.9 332.9,193.8 334.9,194.8 336.9,195.8 338.9,196.7 340.9,197.6 342.9,198.6 344.9,199.5 346.9,200.4 348.9,201.3 350.9,202.2 352.9,203.0 354.8,203.9 356.8,204.8 358.8,205.6 360.8,206.5 362.8,207.3 364.8,208.1 366.8,208.9 368.8,209.7 370.8,210.5 372.8,211.3 374.8,212.1 376.8,212.9 378.8,213.6 380.7,214.4 382.7,215.2 384.7,215.9 386.7,216.6 388.7,217.4 390.7,218.1 392.7,218.8 394.7,219.5 396.7,220.2 398.7,220.9 400.7,221.6 402.7,222.3 404.6,223.0 406.6,223.7 408.6,224.3 410.6,225.0 412.6,225.7 414.6,226.3 416.6,226.9 418.6,227.6 420.6,228.2 422.6,228.9 424.6,229.5 426.6,230.1 428.6,230.7 430.5,231.3 432.5,231.9 434.5,232.5 436.5,233.1 438.5,233.7 440.5,234.3 442.5,234.9 444.5,235.4 446.5,236.0 448.5,236.6 450.5,237.1 452.5,237.7 454.5,238.2 456.4,238.8 458.4,239.3 460.4,239.9 462.4,240.4 464.4,240.9 466.4,241.5 468.4,242.0 470.4,242.5 472.4,243.0 474.4,243.5 476.4,244.1 478.4,244.6 480.4,245.1 482.3,245.6 484.3,246.0 486.3,246.5 488.3,247.0 490.3,247.5 492.3,248.0 494.3,248.5 496.3,248.9 498.3,249.4 500.3,249.9 502.3,250.3 504.3,250.8 506.2,251.2 508.2,251.7 510.2,252.1 512.2,252.6 514.2,253.0 516.2,253.5 518.2,253.9 520.2,254.3 522.2,254.8 524.2,255.2 526.2,255.6 528.2,256.0 530.2,256.5 532.1,256.9 534.1,257.3 536.1,257.7 538.1,258.1 540.1,258.5 542.1,258.9 544.1,259.3 546.1,259.7 548.1,260.1 550.1,260.5 552.1,260.9 554.1,261.3 556.1,261.7 558.0,262.0 560.0,262.4 562.0,262.8 564.0,263.2 566.0,263.5 568.0,263.9 570.0,264.3" class="stroke-c dashed" fill="none"/>
<text x="153.7" y="43.4" class="fig-small" text-anchor="start">F1 = 730</text>
<text x="199.6" y="74.6" class="fig-small" text-anchor="start">F2 = 1090</text>
<text x="371.7" y="149.6" class="fig-small" text-anchor="start">F3 = 2440</text>
<line x1="330" y1="48" x2="360" y2="48" class="stroke-a"/><text x="366" y="52" class="fig-small">frame spectrum (harmonics every 120 Hz)</text>
<line x1="330" y1="66" x2="360" y2="66" class="stroke-b"/><text x="366" y="70" class="fig-small">LPC envelope, order 18</text>
<line x1="330" y1="84" x2="360" y2="84" class="stroke-c dashed"/><text x="366" y="88" class="fig-small">LPC envelope, order 4</text>
</svg>
Figure 1. The orange order-18 LPC envelope, which you build in this chapter, rides over the harmonics and peaks at the three formants; the dashed order-4 envelope has only two pole pairs to spend and blurs F1 and F2 into one broad bump.
```

Linear prediction does this by asking a different question: how well can you guess the next sample from the ones before it? The best guess turns out to be a vocal-tract filter in disguise. This chapter builds that guess, solves for the best one, and reads the formants out of it.

## What you already know

- A resonator is an IIR filter, `scipy.signal.lfilter(b, a, x)`, whose output is a weighted sum of the input and the previous outputs.
- A formant is a pair of complex poles. The pole's angle sets its frequency, and its radius sets its bandwidth.
- A voiced sound is a pulse train, at the pitch, through the vocal-tract filter.
- The autocorrelation of a frame multiplies the frame by a shifted copy of itself and adds.
- A logarithm turns a product into a sum.

## A vowel is predictable from its past

Start with a signal that is trivially predictable, the ramp $x = 1, 2, 3, 4$. Each sample continues the line, so $x[n] = 2x[n-1] - x[n-2]$. Two past samples with the weights $2$ and $-1$ predict every sample after the first two. The *prediction error* $e[n]$ is the sample minus its prediction, and for the ramp it is $[1, 0, 0, 0]$. The 1 is the first sample, which has no past to predict it from. Every other error is 0.

A vowel is predictable for the same reason. A formant resonator computes $y[n] = b_0 x[n] + 2r\cos\theta\, y[n-1] - r^2 y[n-2]$: a fixed weighted sum of its own past, plus the input. The only part of $y[n]$ that the past cannot predict is the input, which is the glottal pulse. Figure 2 shows it: the error is small everywhere except at the pulses.

```figure
<svg viewBox="0 0 600 350" role="img" aria-label="The vowel waveform, and below it on a 25 times larger scale the much smaller prediction error with spikes at each glottal pulse">
<line x1="70" y1="139.0" x2="570" y2="139.0" class="stroke-grid"/>
<text x="64" y="143.0" class="fig-small" text-anchor="end">-0.5</text>
<line x1="70" y1="89.0" x2="570" y2="89.0" class="stroke-grid"/>
<text x="64" y="93.0" class="fig-small" text-anchor="end">0</text>
<line x1="70" y1="39.0" x2="570" y2="39.0" class="stroke-grid"/>
<text x="64" y="43.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70" y1="144" x2="570" y2="144" class="stroke-muted"/>
<line x1="70" y1="34" x2="70" y2="144" class="stroke-muted"/>
<polyline points="70.0,92.5 71.2,106.9 72.5,120.7 73.8,132.0 75.0,138.7 76.2,138.8 77.5,132.1 78.8,119.6 80.0,103.8 81.2,87.2 82.5,71.8 83.8,59.1 85.0,50.1 86.2,45.5 87.5,46.3 88.8,52.3 90.0,62.7 91.2,75.4 92.5,88.0 93.8,98.7 95.0,106.4 96.2,111.1 97.5,112.9 98.8,112.2 100.0,109.2 101.2,104.6 102.5,99.2 103.8,94.1 105.0,90.4 106.2,88.4 107.5,87.7 108.8,87.6 110.0,87.1 111.2,86.1 112.5,84.4 113.8,82.3 115.0,80.0 116.2,77.6 117.5,75.3 118.8,73.8 120.0,73.7 121.2,75.6 122.5,79.8 123.8,85.9 125.0,92.8 126.2,99.6 127.5,105.4 128.8,109.5 130.0,111.6 131.2,111.3 132.5,108.3 133.8,103.0 135.0,95.8 136.2,87.8 137.5,80.2 138.8,74.1 140.0,70.2 141.2,68.5 142.5,69.2 143.8,71.9 145.0,76.3 146.2,81.8 147.5,87.6 148.8,93.1 150.0,97.3 151.2,99.9 152.5,100.9 153.8,100.5 155.0,99.2 156.2,97.2 157.5,95.0 158.8,92.7 160.0,90.7 161.2,89.1 162.5,88.0 163.8,87.4 165.0,87.0 166.2,86.5 167.5,85.8 168.8,84.8 170.0,83.6 171.2,82.7 172.5,82.1 173.8,82.0 175.0,82.5 176.2,83.7 177.5,85.7 178.8,88.4 180.0,91.5 181.2,94.6 182.5,97.3 183.8,99.1 185.0,99.7 186.2,99.2 187.5,97.6 188.8,95.0 190.0,91.8 191.2,88.2 192.5,84.8 193.8,81.9 195.0,79.9 196.2,79.1 197.5,79.5 198.8,80.9 200.0,83.0 201.2,85.5 202.5,88.2 203.8,90.7 205.0,92.8 206.2,94.2 207.5,94.9 208.8,94.9 210.0,94.3 211.2,93.4 212.5,92.2 213.8,91.1 215.0,90.1 216.2,89.2 217.5,88.5 218.8,88.0 220.0,87.5 221.2,87.2 222.5,86.8 223.8,85.7 225.0,82.5 226.2,76.5 227.5,68.1 228.8,59.7 230.0,54.0 231.2,53.2 232.5,57.7 233.8,66.7 235.0,78.7 236.2,92.5 237.5,106.9 238.8,120.7 240.0,132.1 241.2,138.8 242.5,139.0 243.8,132.2 245.0,119.8 246.2,103.9 247.5,87.3 248.8,71.9 250.0,59.1 251.2,50.0 252.5,45.4 253.8,46.2 255.0,52.2 256.2,62.6 257.5,75.3 258.8,87.9 260.0,98.6 261.2,106.4 262.5,111.1 263.8,112.9 265.0,112.2 266.2,109.3 267.5,104.7 268.8,99.3 270.0,94.2 271.2,90.5 272.5,88.5 273.8,87.8 275.0,87.6 276.2,87.2 277.5,86.1 278.8,84.4 280.0,82.3 281.2,80.0 282.5,77.5 283.8,75.2 285.0,73.7 286.2,73.6 287.5,75.6 288.8,79.8 290.0,85.8 291.2,92.8 292.5,99.6 293.8,105.4 295.0,109.6 296.2,111.7 297.5,111.3 298.8,108.4 300.0,103.0 301.2,95.8 302.5,87.8 303.8,80.3 305.0,74.1 306.2,70.1 307.5,68.5 308.8,69.1 310.0,71.8 311.2,76.2 312.5,81.7 313.8,87.6 315.0,93.0 316.2,97.3 317.5,99.9 318.8,100.9 320.0,100.6 321.2,99.2 322.5,97.3 323.8,95.0 325.0,92.8 326.2,90.7 327.5,89.1 328.8,88.1 330.0,87.4 331.2,87.0 332.5,86.5 333.8,85.8 335.0,84.7 336.2,83.6 337.5,82.6 338.8,82.0 340.0,81.9 341.2,82.4 342.5,83.7 343.8,85.7 345.0,88.4 346.2,91.5 347.5,94.6 348.8,97.3 350.0,99.1 351.2,99.8 352.5,99.3 353.8,97.6 355.0,95.0 356.2,91.8 357.5,88.2 358.8,84.8 360.0,81.9 361.2,79.9 362.5,79.1 363.8,79.5 365.0,80.8 366.2,82.9 367.5,85.5 368.8,88.2 370.0,90.7 371.2,92.8 372.5,94.2 373.8,94.9 375.0,94.9 376.2,94.3 377.5,93.4 378.8,92.3 380.0,91.1 381.2,90.1 382.5,89.2 383.8,88.5 385.0,88.0 386.2,87.5 387.5,87.2 388.8,86.8 390.0,86.5 391.2,85.3 392.5,82.2 393.8,76.3 395.0,68.2 396.2,60.1 397.5,54.8 398.8,54.3 400.0,59.0 401.2,68.1 402.5,80.1 403.8,93.6 405.0,107.7 406.2,121.0 407.5,131.8 408.8,138.0 410.0,137.7 411.2,130.7 412.5,118.1 413.8,102.3 415.0,86.0 416.2,71.0 417.5,58.7 418.8,50.1 420.0,46.1 421.2,47.2 422.5,53.4 423.8,63.9 425.0,76.5 426.2,88.9 427.5,99.4 428.8,106.8 430.0,111.2 431.2,112.7 432.5,111.8 433.8,108.8 435.0,104.1 436.2,98.7 437.5,93.7 438.8,90.1 440.0,88.1 441.2,87.5 442.5,87.4 443.8,87.0 445.0,85.9 446.2,84.3 447.5,82.2 448.8,80.0 450.0,77.6 451.2,75.5 452.5,74.1 453.8,74.1 455.0,76.2 456.2,80.4 457.5,86.4 458.8,93.2 460.0,99.9 461.2,105.5 462.5,109.4 463.8,111.3 465.0,110.7 466.2,107.7 467.5,102.2 468.8,95.1 470.0,87.2 471.2,79.9 472.5,74.0 473.8,70.2 475.0,68.8 476.2,69.6 477.5,72.4 478.8,76.8 480.0,82.3 481.2,88.1 482.5,93.4 483.8,97.5 485.0,100.0 486.2,100.8 487.5,100.4 488.8,99.0 490.0,97.0 491.2,94.7 492.5,92.5 493.8,90.5 495.0,88.9 496.2,87.9 497.5,87.3 498.8,86.9 500.0,86.5 501.2,85.7 502.5,84.7 503.8,83.7 505.0,82.8 506.2,82.2 507.5,82.1 508.8,82.7 510.0,84.0 511.2,86.0 512.5,88.6 513.8,91.7 515.0,94.7 516.2,97.3 517.5,99.0 518.8,99.6 520.0,99.0 521.2,97.3 522.5,94.7 523.8,91.5 525.0,87.9 526.2,84.6 527.5,81.8 528.8,79.9 530.0,79.2 531.2,79.7 532.5,81.1 533.8,83.2 535.0,85.8 536.2,88.4 537.5,90.9 538.8,92.9 540.0,94.3 541.2,94.9 542.5,94.8 543.8,94.2 545.0,93.2 546.2,92.1 547.5,91.0 548.8,90.0 550.0,89.1 551.2,88.4 552.5,87.9 553.8,87.5 555.0,87.1 556.2,86.8 557.5,85.7 558.8,82.6 560.0,76.5 561.2,68.2 562.5,59.8 563.8,54.1 565.0,53.3 566.2,57.9 567.5,66.8 568.8,78.8" class="stroke-a" fill="none"/>
<text x="75.0" y="39.0" class="fig-bold" text-anchor="start">the vowel x[n]</text>
<line x1="225.0" y1="34" x2="225.0" y2="144" class="stroke-grid dashed"/>
<line x1="392.5" y1="34" x2="392.5" y2="144" class="stroke-grid dashed"/>
<line x1="558.8" y1="34" x2="558.8" y2="144" class="stroke-grid dashed"/>
<line x1="70" y1="301.0" x2="570" y2="301.0" class="stroke-grid"/>
<text x="64" y="305.0" class="fig-small" text-anchor="end">-0.02</text>
<line x1="70" y1="251.0" x2="570" y2="251.0" class="stroke-grid"/>
<text x="64" y="255.0" class="fig-small" text-anchor="end">0</text>
<line x1="70" y1="201.0" x2="570" y2="201.0" class="stroke-grid"/>
<text x="64" y="205.0" class="fig-small" text-anchor="end">0.02</text>
<line x1="70" y1="306" x2="570" y2="306" class="stroke-muted"/>
<line x1="70" y1="196" x2="70" y2="306" class="stroke-muted"/>
<line x1="70.0" y1="306" x2="70.0" y2="310" class="stroke-muted"/>
<text x="70.0" y="323" class="fig-small" text-anchor="middle">10</text>
<line x1="195.0" y1="306" x2="195.0" y2="310" class="stroke-muted"/>
<text x="195.0" y="323" class="fig-small" text-anchor="middle">110</text>
<line x1="320.0" y1="306" x2="320.0" y2="310" class="stroke-muted"/>
<text x="320.0" y="323" class="fig-small" text-anchor="middle">210</text>
<line x1="445.0" y1="306" x2="445.0" y2="310" class="stroke-muted"/>
<text x="445.0" y="323" class="fig-small" text-anchor="middle">310</text>
<line x1="570.0" y1="306" x2="570.0" y2="310" class="stroke-muted"/>
<text x="570.0" y="323" class="fig-small" text-anchor="middle">410</text>
<text x="320.0" y="340" text-anchor="middle">sample n within the frame (16 kHz, so 133 samples = one pitch period of 8.3 ms)</text>
<polyline points="70.0,256.9 71.2,247.5 72.5,246.1 73.8,250.3 75.0,252.9 76.2,251.5 77.5,249.8 78.8,251.1 80.0,251.1 81.2,251.1 82.5,251.1 83.8,251.1 85.0,251.1 86.2,251.1 87.5,251.1 88.8,251.1 90.0,251.2 91.2,251.3 92.5,251.4 93.8,251.5 95.0,251.4 96.2,251.3 97.5,251.2 98.8,251.0 100.0,250.8 101.2,250.6 102.5,250.4 103.8,250.3 105.0,250.3 106.2,250.4 107.5,250.5 108.8,250.8 110.0,251.0 111.2,251.2 112.5,251.4 113.8,251.6 115.0,251.7 116.2,251.7 117.5,251.6 118.8,251.4 120.0,251.2 121.2,251.0 122.5,250.8 123.8,250.7 125.0,250.6 126.2,250.6 127.5,250.6 128.8,250.7 130.0,250.8 131.2,250.9 132.5,250.9 133.8,251.0 135.0,251.1 136.2,251.1 137.5,251.1 138.8,251.1 140.0,251.1 141.2,251.1 142.5,251.1 143.8,251.1 145.0,251.1 146.2,251.2 147.5,251.2 148.8,251.2 150.0,251.2 151.2,251.1 152.5,251.1 153.8,251.0 155.0,250.9 156.2,250.8 157.5,250.7 158.8,250.7 160.0,250.7 161.2,250.7 162.5,250.8 163.8,250.9 165.0,251.0 166.2,251.1 167.5,251.2 168.8,251.3 170.0,251.3 171.2,251.3 172.5,251.3 173.8,251.2 175.0,251.1 176.2,251.0 177.5,250.9 178.8,250.9 180.0,250.8 181.2,250.8 182.5,250.8 183.8,250.8 185.0,250.9 186.2,250.9 187.5,251.0 188.8,251.0 190.0,251.0 191.2,251.0 192.5,251.0 193.8,251.0 195.0,251.0 196.2,251.0 197.5,251.1 198.8,251.1 200.0,251.1 201.2,251.1 202.5,251.1 203.8,251.1 205.0,251.1 206.2,251.0 207.5,251.0 208.8,251.0 210.0,250.9 211.2,250.9 212.5,250.9 213.8,250.8 215.0,250.9 216.2,250.9 217.5,250.9 218.8,251.0 220.0,251.0 221.2,251.1 222.5,251.1 223.8,232.7 225.0,220.5 226.2,230.9 227.5,250.0 228.8,258.4 230.0,254.5 231.2,248.8 232.5,248.3 233.8,250.9 235.0,252.3 236.2,251.6 237.5,250.5 238.8,250.2 240.0,250.3 241.2,250.6 242.5,250.8 243.8,251.0 245.0,251.1 246.2,251.1 247.5,251.1 248.8,251.1 250.0,251.1 251.2,251.1 252.5,251.1 253.8,251.1 255.0,251.1 256.2,251.2 257.5,251.3 258.8,251.4 260.0,251.5 261.2,251.4 262.5,251.3 263.8,251.2 265.0,251.0 266.2,250.8 267.5,250.6 268.8,250.4 270.0,250.3 271.2,250.3 272.5,250.4 273.8,250.5 275.0,250.8 276.2,251.0 277.5,251.2 278.8,251.4 280.0,251.6 281.2,251.7 282.5,251.7 283.8,251.6 285.0,251.4 286.2,251.2 287.5,251.0 288.8,250.8 290.0,250.7 291.2,250.6 292.5,250.6 293.8,250.6 295.0,250.7 296.2,250.8 297.5,250.9 298.8,250.9 300.0,251.0 301.2,251.1 302.5,251.1 303.8,251.1 305.0,251.1 306.2,251.1 307.5,251.1 308.8,251.1 310.0,251.1 311.2,251.1 312.5,251.2 313.8,251.2 315.0,251.2 316.2,251.2 317.5,251.1 318.8,251.1 320.0,251.0 321.2,250.9 322.5,250.8 323.8,250.7 325.0,250.7 326.2,250.7 327.5,250.7 328.8,250.8 330.0,250.9 331.2,251.0 332.5,251.1 333.8,251.2 335.0,251.3 336.2,251.3 337.5,251.3 338.8,251.3 340.0,251.2 341.2,251.1 342.5,251.0 343.8,250.9 345.0,250.9 346.2,250.8 347.5,250.8 348.8,250.8 350.0,250.8 351.2,250.9 352.5,250.9 353.8,251.0 355.0,251.0 356.2,251.0 357.5,251.0 358.8,251.0 360.0,251.0 361.2,251.0 362.5,251.0 363.8,251.1 365.0,251.1 366.2,251.1 367.5,251.1 368.8,251.1 370.0,251.1 371.2,251.1 372.5,251.0 373.8,251.0 375.0,251.0 376.2,250.9 377.5,250.9 378.8,250.9 380.0,250.8 381.2,250.9 382.5,250.9 383.8,250.9 385.0,251.0 386.2,251.0 387.5,251.1 388.8,251.1 390.0,251.1 391.2,232.7 392.5,220.5 393.8,230.9 395.0,249.9 396.2,258.3 397.5,254.4 398.8,248.8 400.0,248.3 401.2,250.9 402.5,252.3 403.8,251.6 405.0,250.5 406.2,250.2 407.5,250.4 408.8,250.6 410.0,250.8 411.2,251.0 412.5,251.1 413.8,251.1 415.0,251.1 416.2,251.1 417.5,251.1 418.8,251.1 420.0,251.1 421.2,251.1 422.5,251.1 423.8,251.2 425.0,251.3 426.2,251.4 427.5,251.5 428.8,251.4 430.0,251.3 431.2,251.1 432.5,251.0 433.8,250.8 435.0,250.6 436.2,250.4 437.5,250.3 438.8,250.3 440.0,250.4 441.2,250.5 442.5,250.8 443.8,251.0 445.0,251.3 446.2,251.5 447.5,251.6 448.8,251.7 450.0,251.6 451.2,251.6 452.5,251.4 453.8,251.2 455.0,251.0 456.2,250.8 457.5,250.7 458.8,250.6 460.0,250.6 461.2,250.6 462.5,250.7 463.8,250.8 465.0,250.9 466.2,251.0 467.5,251.0 468.8,251.1 470.0,251.1 471.2,251.1 472.5,251.1 473.8,251.1 475.0,251.1 476.2,251.1 477.5,251.1 478.8,251.1 480.0,251.2 481.2,251.2 482.5,251.2 483.8,251.2 485.0,251.1 486.2,251.0 487.5,251.0 488.8,250.9 490.0,250.8 491.2,250.7 492.5,250.7 493.8,250.7 495.0,250.7 496.2,250.8 497.5,250.9 498.8,251.0 500.0,251.1 501.2,251.2 502.5,251.3 503.8,251.3 505.0,251.3 506.2,251.3 507.5,251.2 508.8,251.1 510.0,251.0 511.2,250.9 512.5,250.9 513.8,250.8 515.0,250.8 516.2,250.8 517.5,250.8 518.8,250.9 520.0,250.9 521.2,251.0 522.5,251.0 523.8,251.0 525.0,251.0 526.2,251.0 527.5,251.0 528.8,251.0 530.0,251.0 531.2,251.1 532.5,251.1 533.8,251.1 535.0,251.1 536.2,251.1 537.5,251.1 538.8,251.1 540.0,251.0 541.2,251.0 542.5,251.0 543.8,250.9 545.0,250.9 546.2,250.9 547.5,250.9 548.8,250.9 550.0,250.9 551.2,250.9 552.5,251.0 553.8,251.0 555.0,251.1 556.2,251.1 557.5,232.7 558.8,220.5 560.0,230.9 561.2,250.0 562.5,258.4 563.8,254.5 565.0,248.8 566.2,248.3 567.5,250.9 568.8,252.3" class="stroke-b" fill="none"/>
<text x="75.0" y="201.0" class="fig-bold" text-anchor="start">the prediction error e[n], order 18, on a scale 25 times larger</text>
<line x1="225.0" y1="196" x2="225.0" y2="306" class="stroke-grid dashed"/>
<line x1="392.5" y1="196" x2="392.5" y2="306" class="stroke-grid dashed"/>
<line x1="558.8" y1="196" x2="558.8" y2="306" class="stroke-grid dashed"/>
</svg>
Figure 2. The vowel peaks at 0.50, while the order-18 prediction error never exceeds 0.012, so the bottom panel is zoomed 25 times. Notice that the error is largest at the dashed lines, the glottal pulses, because the past cannot predict a pulse.
```

*Linear prediction* fits the weights. Call $p$ the number of past samples, the *order*, and $\alpha_k$ the weight on $x[n-k]$, stored as `alpha[k - 1]`. Then:

$$
\hat x[n] = \sum_{k=1}^{p} \alpha_k\, x[n-k], \qquad e[n] = x[n] - \hat x[n]
$$

Read it aloud: the prediction $\hat x[n]$ is a weighted sum of the $p$ previous samples, and the error $e[n]$ is what it misses. Filtering $x$ with $A(z) = 1 - \sum_k \alpha_k z^{-k}$ produces $e$, so the reverse filter $1/A(z)$ turns the error back into the vowel. That all-pole filter is the vocal-tract model: the weights *are* the vocal tract, and the error *is* the source.

```python
a = np.concatenate([[1.0], -alpha])          # the polynomial A(z)
e = scipy.signal.lfilter(a, [1.0], x)        # e[n] = x[n] - sum alpha[k-1] x[n-k]
```

> [!KEY]
> If the vocal tract is all-pole, each sample is a fixed weighted sum of the previous $p$ samples plus an input. The weights describe the filter, and the leftover error is the source.

## Choose the weights

Which weights are best? The ones that make the total squared error $\sum_n e[n]^2$ as small as possible. Setting the derivative with respect to each $\alpha_i$ to zero gives $p$ linear equations, the *normal equations*:

$$
\sum_{k=1}^{p} \alpha_k\, r\big[|i - k|\big] = r[i], \quad i = 1, \dots, p
\qquad\text{where}\qquad
r[k] = \sum_{n=k}^{N-1} x[n]\, x[n-k]
$$

Read it aloud: $r[k]$ is the frame's *autocorrelation* at lag $k$ samples, the sum of products of the frame with itself shifted by $k$. The equations say that the weighted autocorrelations must reproduce $r[i]$ for each lag. The matrix has $r[|i-k|]$ in row $i$ and column $k$, so every diagonal is constant, which makes it a *Toeplitz* matrix.

The sums assume the frame is zero outside its $N$ samples, so you multiply it by a Hamming window first. Without the window, the abrupt frame edges look like clicks that the predictor tries to fit.

Take a frame with $r = [1, 0.5, 0.1]$ and $p = 2$. The two equations are $\alpha_1 + 0.5\,\alpha_2 = 0.5$ and $0.5\,\alpha_1 + \alpha_2 = 0.1$. Substituting the first into the second gives $0.75\,\alpha_2 = -0.15$, so $\alpha_2 = -0.2$ and $\alpha_1 = 0.6$.

## Solve it fast with Levinson-Durbin

Why not hand the matrix to `np.linalg.solve`? You get the same answer, but a general solver ignores the Toeplitz structure and takes $O(p^3)$ operations. The *Levinson-Durbin recursion* uses the structure: it solves order 1, builds order 2 from order 1, and continues up to order $p$ in $O(p^2)$.

At step $i$, the order-$(i-1)$ predictor has a leftover error $E_{i-1}$. Part of $r[i]$ is not explained yet, and the *reflection coefficient* $k_i$ measures that part against the error:

$$
k_i = \frac{r[i] - \sum_{j=1}^{i-1} \alpha_j\, r[i-j]}{E_{i-1}}, \qquad
\alpha_i \leftarrow k_i, \qquad
\alpha_j \leftarrow \alpha_j - k_i\, \alpha_{i-j}\ (j < i), \qquad
E_i = (1 - k_i^2)\, E_{i-1}
$$

Read it aloud: divide what the lower-order predictor misses at lag $i$ by its error, and that becomes the last weight. Every older weight is corrected by $k_i$ times its mirror image, and the error shrinks by $1 - k_i^2$. Start with $E_0 = r[0]$.

On the example, with $E_0 = 1$:

1. Order 1: $k_1 = 0.5 / 1 = 0.5$, so $\alpha_1 = 0.5$ and $E_1 = (1 - 0.25) \times 1 = 0.75$.
2. Order 2: $k_2 = (0.1 - 0.5 \times 0.5) / 0.75 = -0.2$, so $\alpha_2 = -0.2$. The old weight is corrected: $\alpha_1 = 0.5 - (-0.2)(0.5) = 0.6$. The error is $E_2 = 0.75 \times (1 - 0.04) = 0.72$.

That matches the substitution: $\alpha = [0.6, -0.2]$. With the autocorrelation method, every $|k_i| < 1$, which guarantees that $1/A(z)$ is stable. In numpy, copy the old weights first, because the update reads them in reverse.

```python
k_i = (r[i] - alpha[: i - 1] @ r[i - 1 : 0 : -1]) / E
old = alpha[: i - 1].copy()
alpha[: i - 1] = old - k_i * old[::-1]
```

> [!KEY]
> Autocorrelation LPC windows a frame, computes $r[0..p]$, and solves the Toeplitz normal equations in $O(p^2)$ with Levinson-Durbin. The final $E$ is the error energy, and $|k_i| < 1$ keeps $1/A(z)$ stable.

## Read the envelope and formants

The all-pole model has a smooth power spectrum, the *envelope*. Its gain is $E = E_p$, the final error energy:

$$
\left|H(e^{j\omega})\right|^2 = \frac{E}{\left|A(e^{j\omega})\right|^2}
$$

Read it aloud: the envelope is the error energy divided by the squared size of $A$ around the unit circle. Where $A$ is small, near a pole, the envelope peaks. Zero-padding the coefficient vector $[1, -\alpha_1, \dots, -\alpha_p]$ to `n_fft` samples and taking its `rfft` evaluates $A$ on the usual frequency grid.

Figure 1 shows order 18 on a synthetic /ɑ/ at 120 Hz with formants set to 730, 1090, and 2440 Hz. The orange envelope ignores the harmonics and peaks at the formants. The dashed order-4 envelope has two pole pairs, and it blurs the first two formants into one bump.



To read the formants as numbers, find the roots of $A(z)$. Each complex root $z$ with a positive angle is one pole. Invert the resonator formulas from the previous chapter:

$$
F = \frac{\angle z}{2\pi}\, f_s, \qquad B = -\frac{f_s}{\pi} \ln |z|
$$

Here $\angle z$ is the root's angle in radians, `np.angle(z)`, $F$ is its frequency in hertz, and $B$ is its bandwidth in hertz. Keep the roots with a bandwidth under about 400 Hz, because broad ones shape the overall tilt rather than a formant. For the example, $A(z) = 1 - 0.6z^{-1} + 0.2z^{-2}$ has roots $0.3 \pm 0.3317j$ with radius 0.4472. At 16 kHz that is $F = 2128$ Hz and $B = 4098$ Hz, so that pole is no formant at all. On the vowel in Figure 1, order 18 finds formants at 721, 1084, and 2439 Hz, with bandwidths of 60, 78, and 149 Hz.

How large should $p$ be? Each formant needs two poles, and speech has about one formant per kilohertz of bandwidth, so $p = f_s / 1000 + 2$ is a good rule. The extra two poles model the glottal pulse and the lips. At 16 kHz, that is 18.

## The cepstrum: a second way to split source from filter

LPC models the filter and throws the source away. The *real cepstrum* keeps both, in different places. It is the inverse DFT of the log magnitude spectrum:

$$
c[n] = \operatorname{IDFT}\!\left(\ln \left|\operatorname{DFT}(x)\right|\right)[n]
$$

Read it aloud: take the spectrum, keep its magnitude, take the natural log, and transform back. The index $n$ is called *quefrency*, and it counts samples. A voiced frame is a source convolved with a filter, so its log spectrum is a *sum*: the log of the smooth envelope plus the log of the harmonic ripple. The smooth part changes slowly with frequency, so it lands at low quefrency, below about 2 ms, which is 32 samples at 16 kHz. The ripple repeats every $F_0$ hertz, so it lands at one peak, at quefrency $f_s / F_0$ samples, the pitch period.

```python
ceps = np.fft.irfft(np.log(np.abs(np.fft.rfft(frame)) + 1e-10), len(frame))
```

To see a repetition become a peak, take 32 samples with $x[0] = 1$ and an echo $x[4] = 0.5$. The cepstrum is 0.2507 at quefrency 4, $-0.0639$ at 8, and 0.0240 at 12, close to the series $a/2$, $-a^2/4$, and $a^3/6$ for an echo of strength $a = 0.5$. An echo shows up at its delay and again at its multiples. A voice repeats each glottal pulse once per period, so at 120 Hz and 16 kHz the peak sits near quefrency 133, as in Figure 3.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Real cepstrum of the vowel frame: vocal tract shape at low quefrency and a pitch peak at 133 samples">
<line x1="60" y1="194.0" x2="570" y2="194.0" class="stroke-grid"/>
<text x="54" y="198.0" class="fig-small" text-anchor="end">0</text>
<line x1="60" y1="144.0" x2="570" y2="144.0" class="stroke-grid"/>
<text x="54" y="148.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="60" y1="94.0" x2="570" y2="94.0" class="stroke-grid"/>
<text x="54" y="98.0" class="fig-small" text-anchor="end">1</text>
<line x1="60" y1="44.0" x2="570" y2="44.0" class="stroke-grid"/>
<text x="54" y="48.0" class="fig-small" text-anchor="end">1.5</text>
<line x1="60" y1="244" x2="570" y2="244" class="stroke-muted"/>
<line x1="60" y1="34" x2="60" y2="244" class="stroke-muted"/>
<line x1="60.0" y1="244" x2="60.0" y2="248" class="stroke-muted"/>
<text x="60.0" y="261" class="fig-small" text-anchor="middle">0</text>
<line x1="125.3" y1="244" x2="125.3" y2="248" class="stroke-muted"/>
<text x="125.3" y="261" class="fig-small" text-anchor="middle">32</text>
<line x1="190.6" y1="244" x2="190.6" y2="248" class="stroke-muted"/>
<text x="190.6" y="261" class="fig-small" text-anchor="middle">64</text>
<line x1="264.0" y1="244" x2="264.0" y2="248" class="stroke-muted"/>
<text x="264.0" y="261" class="fig-small" text-anchor="middle">100</text>
<line x1="331.3" y1="244" x2="331.3" y2="248" class="stroke-muted"/>
<text x="331.3" y="261" class="fig-small" text-anchor="middle">133</text>
<line x1="468.0" y1="244" x2="468.0" y2="248" class="stroke-muted"/>
<text x="468.0" y="261" class="fig-small" text-anchor="middle">200</text>
<line x1="570.0" y1="244" x2="570.0" y2="248" class="stroke-muted"/>
<text x="570.0" y="261" class="fig-small" text-anchor="middle">250</text>
<text x="315.0" y="278" text-anchor="middle">quefrency (samples; 16 samples = 1 ms)</text>
<text x="60" y="26" class="fig-small" text-anchor="start">cepstrum c[n]</text>
<rect x="60.0" y="34" width="65.3" height="210" class="fill-soft-b"/>
<polyline points="62.0,42.9 64.1,126.1 66.1,183.9 68.2,208.0 70.2,205.6 72.2,197.8 74.3,199.2 76.3,208.7 78.4,216.1 80.4,214.4 82.4,205.0 84.5,194.8 86.5,189.6 88.6,190.3 90.6,192.9 92.6,194.0 94.7,193.5 96.7,192.4 98.8,190.9 100.8,190.5 102.8,192.4 104.9,195.1 106.9,195.7 109.0,193.9 111.0,191.9 113.0,191.6 115.1,192.6 117.1,193.6 119.2,194.1 121.2,194.5 123.2,194.8 125.3,194.9 127.3,195.0 129.4,195.6 131.4,196.6 133.4,197.3 135.5,196.8 137.5,195.2 139.6,193.6 141.6,193.2 143.6,193.3 145.7,193.1 147.7,192.3 149.8,191.7 151.8,191.8 153.8,192.4 155.9,192.9 157.9,193.5 160.0,194.2 162.0,194.8 164.0,195.1 166.1,195.0 168.1,195.0 170.2,195.2 172.2,195.2 174.2,194.8 176.3,194.4 178.3,194.0 180.4,193.9 182.4,193.7 184.4,193.5 186.5,193.5 188.5,193.6 190.6,193.7 192.6,193.7 194.6,193.6 196.7,193.8 198.7,194.0 200.8,193.9 202.8,193.7 204.8,193.7 206.9,193.9 208.9,194.1 211.0,193.9 213.0,193.7 215.0,193.9 217.1,194.3 219.1,194.6 221.2,194.5 223.2,194.4 225.2,194.4 227.3,194.6 229.3,194.5 231.4,194.3 233.4,194.1 235.4,194.0 237.5,193.7 239.5,193.2 241.6,192.9 243.6,193.1 245.6,193.5 247.7,193.5 249.7,193.3 251.8,193.6 253.8,194.1 255.8,194.3 257.9,194.3 259.9,194.6 262.0,195.1 264.0,195.2 266.0,194.4 268.1,193.4 270.1,193.2 272.2,194.1 274.2,194.6 276.2,193.2 278.3,190.9 280.3,190.9 282.4,193.6 284.4,195.7 286.4,194.3 288.5,191.9 290.5,193.0 292.6,195.4 294.6,192.8 296.6,192.2 298.7,193.0 300.7,194.5 302.8,195.1 304.8,194.2 306.8,192.9 308.9,192.6 310.9,193.7 313.0,195.3 315.0,196.1 317.0,195.5 319.1,194.2 321.1,192.9 323.2,192.6 325.2,193.9 327.2,194.8 329.3,183.3 331.3,171.5 333.4,172.1 335.4,182.8 337.4,192.6 339.5,194.7 341.5,192.4 343.6,191.3 345.6,192.6 347.6,193.9 349.7,193.5 351.7,192.5 353.8,192.6 355.8,193.9 357.8,194.9 359.9,194.6 361.9,193.4 364.0,192.7 366.0,193.4 368.0,194.9 370.1,194.8 372.1,193.5 374.2,193.1 376.2,194.0 378.2,194.7 380.3,194.5 382.3,194.2 384.4,194.4 386.4,194.5 388.4,194.1 390.5,193.5 392.5,193.5 394.6,193.9 396.6,194.0 398.6,193.7 400.7,193.2 402.7,193.2 404.8,193.5 406.8,193.9 408.8,194.0 410.9,194.1 412.9,194.2 415.0,194.2 417.0,194.1 419.0,194.1 421.1,194.4 423.1,194.6 425.2,194.3 427.2,193.8 429.2,193.6 431.3,193.9 433.3,194.2 435.4,194.1 437.4,193.8 439.4,193.8 441.5,193.8 443.5,193.9 445.6,193.8 447.6,193.9 449.6,194.1 451.7,194.1 453.7,194.0 455.8,193.8 457.8,193.9 459.8,194.3 461.9,194.4 463.9,194.3 466.0,194.1 468.0,194.0 470.0,194.0 472.1,193.9 474.1,193.9 476.2,194.1 478.2,194.1 480.2,193.8 482.3,193.6 484.3,193.7 486.4,194.3 488.4,194.6 490.4,194.4 492.5,194.0 494.5,193.9 496.6,194.2 498.6,194.3 500.6,194.0 502.7,193.7 504.7,193.8 506.8,194.0 508.8,193.7 510.8,193.3 512.9,193.5 514.9,194.5 517.0,195.0 519.0,194.4 521.0,193.6 523.1,193.8 525.1,194.6 527.2,194.8 529.2,194.4 531.2,193.9 533.3,193.7 535.3,193.6 537.4,193.5 539.4,193.9 541.4,194.6 543.5,195.1 545.5,194.8 547.6,194.0 549.6,194.4 551.6,196.7 553.7,198.7 555.7,197.3 557.8,193.7 559.8,193.6 561.8,198.9 563.9,200.3 565.9,190.6 568.0,192.0 570.0,197.3" class="stroke-a" fill="none"/>
<circle cx="331.3" cy="171.5" r="5" class="fill-b"/>
<text x="341.5" y="169.5" class="fig-small" text-anchor="start">peak at n = 133, so F0 = 16000 / 133 = 120 Hz</text>
<text x="129.4" y="49.0" class="fig-small" text-anchor="start">vocal tract: n &lt; 32 (2 ms)</text>
</svg>
Figure 3. The shaded low-quefrency region holds the smooth vocal-tract envelope, and the isolated peak at n = 133 is the pitch period (c[133] = 0.22).
```

> [!KEY]
> You now know how to read a vowel's formants without being fooled by its harmonics. LPC fits the weights that predict each sample from its past, and those weights give an envelope. The cepstrum puts that envelope at low quefrency and the pitch period at a peak.

> [!RECAP]
> - A vowel is nearly predictable from its last $p$ samples, and the prediction error is the glottal source.
> - The best weights solve the Toeplitz normal equations built from $r[0..p]$ of a windowed frame.
> - Levinson-Durbin solves them in $O(p^2)$ with $k_i$ and $E_i = (1 - k_i^2)E_{i-1}$, and $|k_i| < 1$ keeps the filter stable.
> - The envelope is $E / |A|^2$, and formants are narrow-bandwidth roots of $A(z)$.
> - The real cepstrum is $\operatorname{IDFT}(\ln|X|)$, with the vocal tract at low quefrency and the pitch at $f_s / F_0$.

## Build it step by step

The exercise provides `impulse_train`, `resonator`, and `synthesize_vowel` from the previous chapter, so you can run your code on a vowel. The checks test your functions in this order.

1. **`prediction_error(x, alpha)`** returns $e[n]$, the same length as `x`. Filter `x` with `[1, -alpha]` using `lfilter`. The usual mistake is a plus sign on `alpha`.
2. **`autocorrelation(frame, p)`** returns `r[0..p]` as dot products of the frame with shifted copies of itself. The usual mistake is dividing by `N - k`, which can make the matrix indefinite and the filter unstable.
3. **`levinson_durbin(r, p)`** returns `(alpha, E, k)`. The usual mistake is updating `alpha` in place, so later terms read values that are already updated.
4. **`lpc(frame, order)`** multiplies the frame by `np.hamming(len(frame))`, then calls the two functions before it.
5. **`lpc_envelope(alpha, E, n_fft)`** evaluates `E / |A|^2` with `np.fft.rfft`. The usual mistake is dropping the minus signs in `[1, -alpha]`.
6. **`formants(alpha, sample_rate, max_bandwidth, min_frequency)`** calls `np.roots`, keeps positive angles, converts, filters by bandwidth, and sorts.
7. **`real_cepstrum(frame, eps)`** returns `len(frame)` real values. The usual mistake is leaving out `n=len(frame)` in `np.fft.irfft`, which returns one sample short for an odd-length frame.

## Where this breaks

**High voices.** At high $F_0$ the harmonics sit far apart, and LPC fits individual harmonics instead of the envelope. On this chapter's vowel, order 18 finds $F_1$ at 785 Hz for $F_0 = 200$ Hz, which is 7.5% above the true 730 Hz.

**Nasals, tilt, and spurious poles.** In "m" and "n", the nasal cavity adds zeros that an all-pole model can only approximate with extra poles. Real glottal pulses fall off at about 12 dB per octave, so pre-emphasis, from [Convolution and filters](/lessons/009-convolution-and-filters), flattens the tilt before LPC. A too-high order fits noise with extra narrow poles, so filter the roots by bandwidth.

> [!WARNING]
> Sign conventions differ. Some libraries write $A(z) = 1 + \sum a_k z^{-k}$, so $a_k = -\alpha_k$ and the reflection coefficients flip sign. Check which one a reference uses before you compare numbers. Also add a small epsilon before the log in the cepstrum, because $\ln 0 = -\infty$.

## The exercise

You build `prediction_error`, `autocorrelation`, `levinson_durbin`, `lpc`, `lpc_envelope`, `formants`, and `real_cepstrum`. Press **Run** to hear /ɑ/, /i/, and /u/, see each frame's spectrum under its LPC envelope, and read the formants, the prediction gain, and the cepstral pitch that your code recovers.

The checks verify the prediction error on the ramp and on an all-pole signal, compare your autocorrelation with `np.correlate`, and work the Levinson-Durbin example. They compare the recursion with `scipy.linalg.solve_toeplitz` on random stable processes, including $|k_i| < 1$ and $E$ as the true residual energy, and require that `lpc` leaves an error 20 dB below the vowel. They also compare the envelope with `scipy.signal.freqz`, require formants within 5% of 500, 1500, and 2500 Hz, and check the cepstrum against its definition and the echo.
