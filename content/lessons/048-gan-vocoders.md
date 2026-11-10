---
title: GAN vocoders
slug: 048-gan-vocoders
part: "Part 5 — Speech synthesis"
summary: Build the losses, discriminator reshape, and activation of HiFi-GAN and BigVGAN, then train a vocoder on real speech and hear it.
minutes: 60
gpu: true
objectives:
  - Explain why a regression loss averages many valid waveforms into a muffled one, and how a discriminator avoids that.
  - Compute the multi-resolution STFT loss from spectral convergence and log-magnitude distance at three resolutions.
  - Write the least-squares GAN losses and the feature-matching loss, and say which tensor to detach in each step.
  - Reshape a waveform by period for the multi-period discriminator, and explain why its periods are 2, 3, 5, 7, and 11.
  - Implement the Snake activation and its gradient, and explain its periodic bias.
prereqs:
  - 047-wavenet
  - 008-phase-and-griffin-lim
  - 012-hearing-and-the-mel-scale
exercise: 048-gan-vocoders
---

# GAN vocoders

WaveNet sounds natural but runs one sample at a time, far slower than real time. A vocoder that generates every sample at once is easy to build: a stack of transposed convolutions that stretches each mel frame into 256 samples. The hard part is training it, because trained only to match the waveform or the spectrum, it sounds muffled and buzzy. Why does a network that fits the training data so closely sound so dull?

Many waveforms fit one mel spectrogram, and a regression loss averages them. A *generative adversarial network* (GAN) adds a second network, a *discriminator*, trained to tell real audio from generated audio, while the generator is trained to fool it. The discriminator punishes what a spectral loss ignores, such as smeared phase.

HiFi-GAN, from 2020, made this approach standard. In its paper, the V1 model scored a mean opinion score of 4.36 against 4.45 for recordings, at 168 times faster than real time on a V100 GPU. You build its losses, its multi-period discriminator's reshape, and BigVGAN's Snake activation, then train a small HiFi-GAN-style vocoder on one LibriSpeech speaker.

## What you already know

- A vocoder turns a mel spectrogram into samples, as WaveNet does one sample at a time (see [WaveNet](/lessons/047-wavenet)).
- Griffin-Lim recovers a waveform from a magnitude spectrogram by iterating phase estimates (see [Phase and Griffin-Lim](/lessons/008-phase-and-griffin-lim)).
- An 80-band log-mel spectrogram at a hop of 256 samples is the input most vocoders take (see [Hearing and the mel scale](/lessons/012-hearing-and-the-mel-scale)).
- An STFT frame's magnitude says how much of each frequency it holds, and its phase says where each sinusoid is in its cycle.
- The Frobenius norm of a matrix is the square root of the sum of its squared entries.

## Why regression sounds muffled

Take one tone. A sine wave and the same sine shifted by 120 degrees have the same magnitude spectrum, one unit-amplitude peak, so both answer a mel spectrogram with that peak. Squared error can't pick one, because the training data contains both. The value that minimizes the squared error is their average, which Figure 1 shows has amplitude $\cos(60^\circ) = 0.5$. It matches neither.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Two sine waves with the same magnitude spectrum and their average, whose amplitude is only 0.5">
<line x1="60.0" y1="180.0" x2="550.0" y2="180.0" class="stroke-muted"/>
<line x1="60.0" y1="85.0" x2="60.0" y2="275.0" class="stroke-grid"/>
<line x1="305.0" y1="85.0" x2="305.0" y2="275.0" class="stroke-grid"/>
<line x1="550.0" y1="85.0" x2="550.0" y2="275.0" class="stroke-grid"/>
<polyline points="60.0,180.0 61.2,177.3 62.5,174.7 63.7,172.0 64.9,169.3 66.1,166.7 67.3,164.1 68.6,161.5 69.8,158.9 71.0,156.3 72.2,153.7 73.5,151.2 74.7,148.7 75.9,146.2 77.2,143.8 78.4,141.4 79.6,139.1 80.8,136.7 82.0,134.5 83.3,132.2 84.5,130.0 85.7,127.9 87.0,125.8 88.2,123.8 89.4,121.8 90.6,119.9 91.8,118.0 93.1,116.2 94.3,114.5 95.5,112.8 96.8,111.2 98.0,109.7 99.2,108.2 100.4,106.8 101.7,105.5 102.9,104.3 104.1,103.1 105.3,102.0 106.5,101.0 107.8,100.0 109.0,99.2 110.2,98.4 111.5,97.7 112.7,97.0 113.9,96.5 115.1,96.0 116.3,95.7 117.6,95.4 118.8,95.2 120.0,95.0 121.2,95.0 122.5,95.0 123.7,95.2 124.9,95.4 126.2,95.7 127.4,96.0 128.6,96.5 129.8,97.0 131.1,97.7 132.3,98.4 133.5,99.2 134.7,100.0 135.9,101.0 137.2,102.0 138.4,103.1 139.6,104.3 140.8,105.5 142.1,106.8 143.3,108.2 144.5,109.7 145.8,111.2 147.0,112.8 148.2,114.5 149.4,116.2 150.7,118.0 151.9,119.9 153.1,121.8 154.3,123.8 155.6,125.8 156.8,127.9 158.0,130.0 159.2,132.2 160.4,134.5 161.7,136.7 162.9,139.1 164.1,141.4 165.3,143.8 166.6,146.2 167.8,148.7 169.0,151.2 170.2,153.7 171.5,156.3 172.7,158.9 173.9,161.5 175.2,164.1 176.4,166.7 177.6,169.3 178.8,172.0 180.1,174.7 181.3,177.3 182.5,180.0 183.7,182.7 184.9,185.3 186.2,188.0 187.4,190.7 188.6,193.3 189.8,195.9 191.1,198.5 192.3,201.1 193.5,203.7 194.8,206.3 196.0,208.8 197.2,211.3 198.4,213.8 199.7,216.2 200.9,218.6 202.1,220.9 203.3,223.3 204.6,225.5 205.8,227.8 207.0,230.0 208.2,232.1 209.4,234.2 210.7,236.2 211.9,238.2 213.1,240.1 214.3,242.0 215.6,243.8 216.8,245.5 218.0,247.2 219.2,248.8 220.5,250.3 221.7,251.8 222.9,253.2 224.2,254.5 225.4,255.7 226.6,256.9 227.8,258.0 229.1,259.0 230.3,260.0 231.5,260.8 232.7,261.6 233.9,262.3 235.2,263.0 236.4,263.5 237.6,264.0 238.8,264.3 240.1,264.6 241.3,264.8 242.5,265.0 243.8,265.0 245.0,265.0 246.2,264.8 247.4,264.6 248.7,264.3 249.9,264.0 251.1,263.5 252.3,263.0 253.6,262.3 254.8,261.6 256.0,260.8 257.2,260.0 258.4,259.0 259.7,258.0 260.9,256.9 262.1,255.7 263.4,254.5 264.6,253.2 265.8,251.8 267.0,250.3 268.2,248.8 269.5,247.2 270.7,245.5 271.9,243.8 273.1,242.0 274.4,240.1 275.6,238.2 276.8,236.2 278.1,234.2 279.3,232.1 280.5,230.0 281.7,227.8 282.9,225.5 284.2,223.3 285.4,220.9 286.6,218.6 287.9,216.2 289.1,213.8 290.3,211.3 291.5,208.8 292.8,206.3 294.0,203.7 295.2,201.1 296.4,198.5 297.6,195.9 298.9,193.3 300.1,190.7 301.3,188.0 302.6,185.3 303.8,182.7 305.0,180.0 306.2,177.3 307.4,174.7 308.7,172.0 309.9,169.3 311.1,166.7 312.4,164.1 313.6,161.5 314.8,158.9 316.0,156.3 317.2,153.7 318.5,151.2 319.7,148.7 320.9,146.2 322.1,143.8 323.4,141.4 324.6,139.1 325.8,136.7 327.1,134.5 328.3,132.2 329.5,130.0 330.7,127.9 331.9,125.8 333.2,123.8 334.4,121.8 335.6,119.9 336.9,118.0 338.1,116.2 339.3,114.5 340.5,112.8 341.8,111.2 343.0,109.7 344.2,108.2 345.4,106.8 346.6,105.5 347.9,104.3 349.1,103.1 350.3,102.0 351.6,101.0 352.8,100.0 354.0,99.2 355.2,98.4 356.4,97.7 357.7,97.0 358.9,96.5 360.1,96.0 361.4,95.7 362.6,95.4 363.8,95.2 365.0,95.0 366.2,95.0 367.5,95.0 368.7,95.2 369.9,95.4 371.1,95.7 372.4,96.0 373.6,96.5 374.8,97.0 376.1,97.7 377.3,98.4 378.5,99.2 379.7,100.0 380.9,101.0 382.2,102.0 383.4,103.1 384.6,104.3 385.9,105.5 387.1,106.8 388.3,108.2 389.5,109.7 390.8,111.2 392.0,112.8 393.2,114.5 394.4,116.2 395.6,118.0 396.9,119.9 398.1,121.8 399.3,123.8 400.6,125.8 401.8,127.9 403.0,130.0 404.2,132.2 405.4,134.5 406.7,136.7 407.9,139.1 409.1,141.4 410.4,143.8 411.6,146.2 412.8,148.7 414.0,151.2 415.2,153.7 416.5,156.3 417.7,158.9 418.9,161.5 420.1,164.1 421.4,166.7 422.6,169.3 423.8,172.0 425.1,174.7 426.3,177.3 427.5,180.0 428.7,182.7 429.9,185.3 431.2,188.0 432.4,190.7 433.6,193.3 434.9,195.9 436.1,198.5 437.3,201.1 438.5,203.7 439.8,206.3 441.0,208.8 442.2,211.3 443.4,213.8 444.6,216.2 445.9,218.6 447.1,220.9 448.3,223.3 449.6,225.5 450.8,227.8 452.0,230.0 453.2,232.1 454.4,234.2 455.7,236.2 456.9,238.2 458.1,240.1 459.4,242.0 460.6,243.8 461.8,245.5 463.0,247.2 464.2,248.8 465.5,250.3 466.7,251.8 467.9,253.2 469.1,254.5 470.4,255.7 471.6,256.9 472.8,258.0 474.1,259.0 475.3,260.0 476.5,260.8 477.7,261.6 478.9,262.3 480.2,263.0 481.4,263.5 482.6,264.0 483.9,264.3 485.1,264.6 486.3,264.8 487.5,265.0 488.8,265.0 490.0,265.0 491.2,264.8 492.4,264.6 493.6,264.3 494.9,264.0 496.1,263.5 497.3,263.0 498.6,262.3 499.8,261.6 501.0,260.8 502.2,260.0 503.4,259.0 504.7,258.0 505.9,256.9 507.1,255.7 508.4,254.5 509.6,253.2 510.8,251.8 512.0,250.3 513.2,248.8 514.5,247.2 515.7,245.5 516.9,243.8 518.1,242.0 519.4,240.1 520.6,238.2 521.8,236.2 523.0,234.2 524.3,232.1 525.5,230.0 526.7,227.8 528.0,225.5 529.2,223.3 530.4,220.9 531.6,218.6 532.9,216.2 534.1,213.8 535.3,211.3 536.5,208.8 537.8,206.3 539.0,203.7 540.2,201.1 541.4,198.5 542.6,195.9 543.9,193.3 545.1,190.7 546.3,188.0 547.5,185.3 548.8,182.7 550.0,180.0" class="stroke-a" fill="none"/>
<polyline points="60.0,106.4 61.2,107.8 62.5,109.2 63.7,110.7 64.9,112.3 66.1,113.9 67.3,115.7 68.6,117.4 69.8,119.3 71.0,121.2 72.2,123.1 73.5,125.1 74.7,127.2 75.9,129.3 77.2,131.5 78.4,133.7 79.6,136.0 80.8,138.3 82.0,140.6 83.3,143.0 84.5,145.4 85.7,147.9 87.0,150.4 88.2,152.9 89.4,155.4 90.6,158.0 91.8,160.6 93.1,163.2 94.3,165.8 95.5,168.5 96.8,171.1 98.0,173.8 99.2,176.4 100.4,179.1 101.7,181.8 102.9,184.4 104.1,187.1 105.3,189.8 106.5,192.4 107.8,195.1 109.0,197.7 110.2,200.3 111.5,202.9 112.7,205.4 113.9,208.0 115.1,210.5 116.3,212.9 117.6,215.4 118.8,217.8 120.0,220.2 121.2,222.5 122.5,224.8 123.7,227.0 124.9,229.2 126.2,231.4 127.4,233.5 128.6,235.5 129.8,237.5 131.1,239.5 132.3,241.3 133.5,243.2 134.7,244.9 135.9,246.6 137.2,248.2 138.4,249.8 139.6,251.3 140.8,252.7 142.1,254.1 143.3,255.3 144.5,256.5 145.8,257.7 147.0,258.7 148.2,259.7 149.4,260.6 150.7,261.4 151.9,262.1 153.1,262.8 154.3,263.3 155.6,263.8 156.8,264.2 158.0,264.5 159.2,264.8 160.4,264.9 161.7,265.0 162.9,265.0 164.1,264.9 165.3,264.7 166.6,264.4 167.8,264.1 169.0,263.7 170.2,263.1 171.5,262.5 172.7,261.9 173.9,261.1 175.2,260.3 176.4,259.4 177.6,258.4 178.8,257.3 180.1,256.1 181.3,254.9 182.5,253.6 183.7,252.2 184.9,250.8 186.2,249.3 187.4,247.7 188.6,246.1 189.8,244.3 191.1,242.6 192.3,240.7 193.5,238.8 194.8,236.9 196.0,234.9 197.2,232.8 198.4,230.7 199.7,228.5 200.9,226.3 202.1,224.0 203.3,221.7 204.6,219.4 205.8,217.0 207.0,214.6 208.2,212.1 209.4,209.6 210.7,207.1 211.9,204.6 213.1,202.0 214.3,199.4 215.6,196.8 216.8,194.2 218.0,191.5 219.2,188.9 220.5,186.2 221.7,183.6 222.9,180.9 224.2,178.2 225.4,175.6 226.6,172.9 227.8,170.2 229.1,167.6 230.3,164.9 231.5,162.3 232.7,159.7 233.9,157.1 235.2,154.6 236.4,152.0 237.6,149.5 238.8,147.1 240.1,144.6 241.3,142.2 242.5,139.8 243.8,137.5 245.0,135.2 246.2,133.0 247.4,130.8 248.7,128.6 249.9,126.5 251.1,124.5 252.3,122.5 253.6,120.5 254.8,118.7 256.0,116.8 257.2,115.1 258.4,113.4 259.7,111.8 260.9,110.2 262.1,108.7 263.4,107.3 264.6,105.9 265.8,104.7 267.0,103.5 268.2,102.3 269.5,101.3 270.7,100.3 271.9,99.4 273.1,98.6 274.4,97.9 275.6,97.2 276.8,96.7 278.1,96.2 279.3,95.8 280.5,95.5 281.7,95.2 282.9,95.1 284.2,95.0 285.4,95.0 286.6,95.1 287.9,95.3 289.1,95.6 290.3,95.9 291.5,96.3 292.8,96.9 294.0,97.5 295.2,98.1 296.4,98.9 297.6,99.7 298.9,100.6 300.1,101.6 301.3,102.7 302.6,103.9 303.8,105.1 305.0,106.4 306.2,107.8 307.4,109.2 308.7,110.7 309.9,112.3 311.1,113.9 312.4,115.7 313.6,117.4 314.8,119.3 316.0,121.2 317.2,123.1 318.5,125.1 319.7,127.2 320.9,129.3 322.1,131.5 323.4,133.7 324.6,136.0 325.8,138.3 327.1,140.6 328.3,143.0 329.5,145.4 330.7,147.9 331.9,150.4 333.2,152.9 334.4,155.4 335.6,158.0 336.9,160.6 338.1,163.2 339.3,165.8 340.5,168.5 341.8,171.1 343.0,173.8 344.2,176.4 345.4,179.1 346.6,181.8 347.9,184.4 349.1,187.1 350.3,189.8 351.6,192.4 352.8,195.1 354.0,197.7 355.2,200.3 356.4,202.9 357.7,205.4 358.9,208.0 360.1,210.5 361.4,212.9 362.6,215.4 363.8,217.8 365.0,220.2 366.2,222.5 367.5,224.8 368.7,227.0 369.9,229.2 371.1,231.4 372.4,233.5 373.6,235.5 374.8,237.5 376.1,239.5 377.3,241.3 378.5,243.2 379.7,244.9 380.9,246.6 382.2,248.2 383.4,249.8 384.6,251.3 385.9,252.7 387.1,254.1 388.3,255.3 389.5,256.5 390.8,257.7 392.0,258.7 393.2,259.7 394.4,260.6 395.6,261.4 396.9,262.1 398.1,262.8 399.3,263.3 400.6,263.8 401.8,264.2 403.0,264.5 404.2,264.8 405.4,264.9 406.7,265.0 407.9,265.0 409.1,264.9 410.4,264.7 411.6,264.4 412.8,264.1 414.0,263.7 415.2,263.1 416.5,262.5 417.7,261.9 418.9,261.1 420.1,260.3 421.4,259.4 422.6,258.4 423.8,257.3 425.1,256.1 426.3,254.9 427.5,253.6 428.7,252.2 429.9,250.8 431.2,249.3 432.4,247.7 433.6,246.1 434.9,244.3 436.1,242.6 437.3,240.7 438.5,238.8 439.8,236.9 441.0,234.9 442.2,232.8 443.4,230.7 444.6,228.5 445.9,226.3 447.1,224.0 448.3,221.7 449.6,219.4 450.8,217.0 452.0,214.6 453.2,212.1 454.4,209.6 455.7,207.1 456.9,204.6 458.1,202.0 459.4,199.4 460.6,196.8 461.8,194.2 463.0,191.5 464.2,188.9 465.5,186.2 466.7,183.6 467.9,180.9 469.1,178.2 470.4,175.6 471.6,172.9 472.8,170.2 474.1,167.6 475.3,164.9 476.5,162.3 477.7,159.7 478.9,157.1 480.2,154.6 481.4,152.0 482.6,149.5 483.9,147.1 485.1,144.6 486.3,142.2 487.5,139.8 488.8,137.5 490.0,135.2 491.2,133.0 492.4,130.8 493.6,128.6 494.9,126.5 496.1,124.5 497.3,122.5 498.6,120.5 499.8,118.7 501.0,116.8 502.2,115.1 503.4,113.4 504.7,111.8 505.9,110.2 507.1,108.7 508.4,107.3 509.6,105.9 510.8,104.7 512.0,103.5 513.2,102.3 514.5,101.3 515.7,100.3 516.9,99.4 518.1,98.6 519.4,97.9 520.6,97.2 521.8,96.7 523.0,96.2 524.3,95.8 525.5,95.5 526.7,95.2 528.0,95.1 529.2,95.0 530.4,95.0 531.6,95.1 532.9,95.3 534.1,95.6 535.3,95.9 536.5,96.3 537.8,96.9 539.0,97.5 540.2,98.1 541.4,98.9 542.6,99.7 543.9,100.6 545.1,101.6 546.3,102.7 547.5,103.9 548.8,105.1 550.0,106.4" class="stroke-c" fill="none"/>
<polyline points="60.0,143.2 61.2,142.5 62.5,141.9 63.7,141.4 64.9,140.8 66.1,140.3 67.3,139.9 68.6,139.4 69.8,139.1 71.0,138.7 72.2,138.4 73.5,138.2 74.7,138.0 75.9,137.8 77.2,137.6 78.4,137.6 79.6,137.5 80.8,137.5 82.0,137.5 83.3,137.6 84.5,137.7 85.7,137.9 87.0,138.1 88.2,138.3 89.4,138.6 90.6,138.9 91.8,139.3 93.1,139.7 94.3,140.2 95.5,140.7 96.8,141.2 98.0,141.7 99.2,142.3 100.4,143.0 101.7,143.6 102.9,144.4 104.1,145.1 105.3,145.9 106.5,146.7 107.8,147.5 109.0,148.4 110.2,149.3 111.5,150.3 112.7,151.2 113.9,152.2 115.1,153.3 116.3,154.3 117.6,155.4 118.8,156.5 120.0,157.6 121.2,158.8 122.5,159.9 123.7,161.1 124.9,162.3 126.2,163.5 127.4,164.8 128.6,166.0 129.8,167.3 131.1,168.6 132.3,169.9 133.5,171.2 134.7,172.5 135.9,173.8 137.2,175.1 138.4,176.4 139.6,177.8 140.8,179.1 142.1,180.4 143.3,181.8 144.5,183.1 145.8,184.4 147.0,185.8 148.2,187.1 149.4,188.4 150.7,189.7 151.9,191.0 153.1,192.3 154.3,193.6 155.6,194.8 156.8,196.1 158.0,197.3 159.2,198.5 160.4,199.7 161.7,200.9 162.9,202.0 164.1,203.1 165.3,204.3 166.6,205.3 167.8,206.4 169.0,207.4 170.2,208.4 171.5,209.4 172.7,210.4 173.9,211.3 175.2,212.2 176.4,213.0 177.6,213.9 178.8,214.6 180.1,215.4 181.3,216.1 182.5,216.8 183.7,217.5 184.9,218.1 186.2,218.6 187.4,219.2 188.6,219.7 189.8,220.1 191.1,220.6 192.3,220.9 193.5,221.3 194.8,221.6 196.0,221.8 197.2,222.0 198.4,222.2 199.7,222.4 200.9,222.4 202.1,222.5 203.3,222.5 204.6,222.5 205.8,222.4 207.0,222.3 208.2,222.1 209.4,221.9 210.7,221.7 211.9,221.4 213.1,221.1 214.3,220.7 215.6,220.3 216.8,219.8 218.0,219.3 219.2,218.8 220.5,218.3 221.7,217.7 222.9,217.0 224.2,216.4 225.4,215.6 226.6,214.9 227.8,214.1 229.1,213.3 230.3,212.5 231.5,211.6 232.7,210.7 233.9,209.7 235.2,208.8 236.4,207.8 237.6,206.7 238.8,205.7 240.1,204.6 241.3,203.5 242.5,202.4 243.8,201.3 245.0,200.1 246.2,198.9 247.4,197.7 248.7,196.5 249.9,195.2 251.1,194.0 252.3,192.7 253.6,191.4 254.8,190.1 256.0,188.8 257.2,187.5 258.4,186.2 259.7,184.9 260.9,183.6 262.1,182.2 263.4,180.9 264.6,179.6 265.8,178.2 267.0,176.9 268.2,175.6 269.5,174.2 270.7,172.9 271.9,171.6 273.1,170.3 274.4,169.0 275.6,167.7 276.8,166.4 278.1,165.2 279.3,163.9 280.5,162.7 281.7,161.5 282.9,160.3 284.2,159.1 285.4,158.0 286.6,156.9 287.9,155.7 289.1,154.7 290.3,153.6 291.5,152.6 292.8,151.6 294.0,150.6 295.2,149.6 296.4,148.7 297.6,147.8 298.9,147.0 300.1,146.1 301.3,145.4 302.6,144.6 303.8,143.9 305.0,143.2 306.2,142.5 307.4,141.9 308.7,141.4 309.9,140.8 311.1,140.3 312.4,139.9 313.6,139.4 314.8,139.1 316.0,138.7 317.2,138.4 318.5,138.2 319.7,138.0 320.9,137.8 322.1,137.6 323.4,137.6 324.6,137.5 325.8,137.5 327.1,137.5 328.3,137.6 329.5,137.7 330.7,137.9 331.9,138.1 333.2,138.3 334.4,138.6 335.6,138.9 336.9,139.3 338.1,139.7 339.3,140.2 340.5,140.7 341.8,141.2 343.0,141.7 344.2,142.3 345.4,143.0 346.6,143.6 347.9,144.4 349.1,145.1 350.3,145.9 351.6,146.7 352.8,147.5 354.0,148.4 355.2,149.3 356.4,150.3 357.7,151.2 358.9,152.2 360.1,153.3 361.4,154.3 362.6,155.4 363.8,156.5 365.0,157.6 366.2,158.7 367.5,159.9 368.7,161.1 369.9,162.3 371.1,163.5 372.4,164.8 373.6,166.0 374.8,167.3 376.1,168.6 377.3,169.9 378.5,171.2 379.7,172.5 380.9,173.8 382.2,175.1 383.4,176.4 384.6,177.8 385.9,179.1 387.1,180.4 388.3,181.8 389.5,183.1 390.8,184.4 392.0,185.8 393.2,187.1 394.4,188.4 395.6,189.7 396.9,191.0 398.1,192.3 399.3,193.6 400.6,194.8 401.8,196.1 403.0,197.3 404.2,198.5 405.4,199.7 406.7,200.9 407.9,202.0 409.1,203.1 410.4,204.3 411.6,205.3 412.8,206.4 414.0,207.4 415.2,208.4 416.5,209.4 417.7,210.4 418.9,211.3 420.1,212.2 421.4,213.0 422.6,213.9 423.8,214.6 425.1,215.4 426.3,216.1 427.5,216.8 428.7,217.5 429.9,218.1 431.2,218.6 432.4,219.2 433.6,219.7 434.9,220.1 436.1,220.6 437.3,220.9 438.5,221.3 439.8,221.6 441.0,221.8 442.2,222.0 443.4,222.2 444.6,222.4 445.9,222.4 447.1,222.5 448.3,222.5 449.6,222.5 450.8,222.4 452.0,222.3 453.2,222.1 454.4,221.9 455.7,221.7 456.9,221.4 458.1,221.1 459.4,220.7 460.6,220.3 461.8,219.8 463.0,219.3 464.2,218.8 465.5,218.3 466.7,217.7 467.9,217.0 469.1,216.4 470.4,215.6 471.6,214.9 472.8,214.1 474.1,213.3 475.3,212.5 476.5,211.6 477.7,210.7 478.9,209.7 480.2,208.8 481.4,207.8 482.6,206.7 483.9,205.7 485.1,204.6 486.3,203.5 487.5,202.4 488.8,201.3 490.0,200.1 491.2,198.9 492.4,197.7 493.6,196.5 494.9,195.2 496.1,194.0 497.3,192.7 498.6,191.4 499.8,190.1 501.0,188.8 502.2,187.5 503.4,186.2 504.7,184.9 505.9,183.6 507.1,182.2 508.4,180.9 509.6,179.6 510.8,178.2 512.0,176.9 513.2,175.6 514.5,174.2 515.7,172.9 516.9,171.6 518.1,170.3 519.4,169.0 520.6,167.7 521.8,166.4 523.0,165.2 524.3,163.9 525.5,162.7 526.7,161.5 528.0,160.3 529.2,159.1 530.4,158.0 531.6,156.9 532.9,155.7 534.1,154.7 535.3,153.6 536.5,152.6 537.8,151.6 539.0,150.6 540.2,149.6 541.4,148.7 542.6,147.8 543.9,147.0 545.1,146.1 546.3,145.4 547.5,144.6 548.8,143.9 550.0,143.2" class="stroke-b" fill="none" stroke-width="3"/>
<text x="60.0" y="30.0" class="fig-small" text-anchor="start">valid waveform 1: sin(t)</text>
<line x1="30.0" y1="26.0" x2="54.0" y2="26.0" class="stroke-a"/>
<text x="260.0" y="30.0" class="fig-small" text-anchor="start">valid waveform 2: sin(t + 120 degrees)</text>
<line x1="230.0" y1="26.0" x2="254.0" y2="26.0" class="stroke-c"/>
<text x="60.0" y="52.0" class="fig-small" text-anchor="start">their average, the regression answer: amplitude 0.5</text>
<line x1="30.0" y1="48.0" x2="54.0" y2="48.0" class="stroke-b" stroke-width="3"/>
<text x="305.0" y="300.0" class="fig-small" text-anchor="middle">time (two periods of the tone)</text>
<text x="52.0" y="99.0" class="fig-small" text-anchor="end">1</text>
<text x="52.0" y="269.0" class="fig-small" text-anchor="end">-1</text>
</svg>
Figure 1. The two waveforms have the same spectrum magnitude, since each is one unit-amplitude tone, but their average has amplitude cos(60 degrees) = 0.5. Notice that the regression answer matches neither waveform and has half the right spectrum magnitude.
```

Real speech has this problem at every frequency in every frame, so a regression vocoder sounds smeared, and a better regression loss doesn't help. You need a judge that asks "does this look like a real waveform?" and doesn't care which one.

## A critic and a generator

A GAN is that judge. HiFi-GAN uses the *least-squares GAN* (LSGAN) objective. The discriminator $D$ is a regressor trained to output 1 on real audio $x$ and 0 on generated audio $G(s)$ from the mel $s$. The generator is trained to make $D$ output 1 on its audio:

$$
\mathcal{L}_{\text{adv}}(D; G) = \mathbb{E}\left[(D(x) - 1)^2 + D(G(s))^2\right], \qquad
\mathcal{L}_{\text{adv}}(G; D) = \mathbb{E}\left[(D(G(s)) - 1)^2\right].
$$

Read it aloud: the discriminator pays the squared distance of its real score from 1 and its fake score from 0, and the generator pays the squared distance of the fake score from 1. Each sub-discriminator outputs a map of scores, the expectation is the mean over it, and the losses add over sub-discriminators. For one score, $D(x) = 0.9$ and $D(G(s)) = 0.3$ give $\mathcal{L}_D = 0.01 + 0.09 = 0.10$ and $\mathcal{L}_G = 0.49$. The generator's gradient with respect to that score is $2(0.3 - 1) = -1.4$: raise the score.

*Feature matching* adds a denser signal: the generator must produce the same discriminator features on its audio as on the real audio, at every layer:

$$
\mathcal{L}_{\text{FM}}(G; D) = \mathbb{E}\left[\sum_{i} \frac{1}{N_i} \big\lVert D^i(x) - D^i(G(s)) \big\rVert_1\right].
$$

Read it aloud: at each layer $i$, take the mean absolute difference over its $N_i$ features, and add over the layers. Real features $[1, 2]$ against generated $[1.5, 1]$ give $(0.5 + 1)/2 = 0.75$ for that layer. HiFi-GAN's generator loss adds the adversarial term, feature matching with weight 2, and an L1 distance between the real and generated log-mel with weight 45. The run here also adds the STFT loss below with weight 2.5.

> [!WARNING]
> Detach $G(s)$ in the discriminator's step, or the discriminator's loss trains the generator to be caught. Don't detach it in the generator's step, or the generator gets no adversarial gradient.

## Anchor the spectrum

An adversarial loss alone trains slowly and wanders, so a vocoder also needs a loss that says "the spectrum is right." One STFT resolution isn't enough: a long window checks harmonics and misses the timing of a click, and a short window does the reverse. Parallel WaveGAN's *multi-resolution STFT loss* checks three resolutions, combining two terms at each:

$$
L_{\text{sc}} = \frac{\big\lVert\, |S| - |\hat S| \,\big\rVert_F}{\big\lVert\, |S| \,\big\rVert_F}, \qquad
L_{\text{mag}} = \frac{1}{N} \big\lVert \log |S| - \log |\hat S| \big\rVert_1.
$$

Read it aloud: $|S|$ and $|\hat S|$ are the STFT magnitudes of the real and generated audio. *Spectral convergence* $L_{\text{sc}}$ is the magnitude error relative to the real magnitude, so loud bins dominate it. $L_{\text{mag}}$ is the mean absolute difference of log magnitudes over all $N$ entries, so a quiet bin counts as much as a loud one. The loss averages $L_{\text{sc}} + L_{\text{mag}}$ over the three resolutions, which use FFT sizes 1024, 2048, and 512, window lengths 600, 1200, and 240, and hops 120, 240, and 50, all with Hann windows. For one resolution in torch:

```python
mag = torch.stft(x, n_fft, hop, win, window=torch.hann_window(win),
                 return_complex=True).abs().clamp(min=1e-7)
sc = torch.linalg.norm(mag - mag_hat) / torch.linalg.norm(mag)
```

Two small examples show the difference. If $|S|$ has Frobenius norm 10 and the error has norm 2, then $L_{\text{sc}} = 0.2$. If every generated magnitude is $e^{0.1}$ times the real one, $L_{\text{mag}} = 0.1$, loud or quiet. Both terms are unchanged when you scale both signals by the same gain.

> [!KEY]
> The adversarial and feature-matching losses make the audio sound real. The multi-resolution STFT loss makes its spectrum right at three window lengths.

## Judge the periodicity

Voiced speech is periodic: a 200 Hz vowel at 16 kHz repeats every 80 samples. HiFi-GAN's *multi-period discriminator* (MPD) judges that periodicity. Each of its five sub-discriminators takes a period $p$ and folds the waveform of length $T$ into a 2-D array of $\lceil T/p \rceil$ rows and $p$ columns (Figure 2), then applies 2-D convolutions whose kernels are 1 wide along the columns:

$$
X^{(p)}_{r, c} = x_{r p + c}, \qquad r = 0, \ldots, \lceil T/p \rceil - 1, \quad c = 0, \ldots, p - 1.
$$

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Ten samples folded with period 4 into three rows of four, with the last row padded by reflection; column 1 holds samples 1, 5, and 9">
<text x="20.0" y="30.0" class="fig-bold" text-anchor="start">waveform, 10 samples</text>
<rect x="20.0" y="42.0" width="32.0" height="28.0" class="fill-soft-a"/>
<rect x="20.0" y="42.0" width="32.0" height="28.0" class="stroke-a" fill="none"/>
<text x="36.0" y="61.0" class="fig-small" text-anchor="middle">0</text>
<rect x="56.0" y="42.0" width="32.0" height="28.0" class="fill-soft-a"/>
<rect x="56.0" y="42.0" width="32.0" height="28.0" class="stroke-a" fill="none"/>
<text x="72.0" y="61.0" class="fig-small" text-anchor="middle">1</text>
<rect x="92.0" y="42.0" width="32.0" height="28.0" class="fill-soft-a"/>
<rect x="92.0" y="42.0" width="32.0" height="28.0" class="stroke-a" fill="none"/>
<text x="108.0" y="61.0" class="fig-small" text-anchor="middle">2</text>
<rect x="128.0" y="42.0" width="32.0" height="28.0" class="fill-soft-a"/>
<rect x="128.0" y="42.0" width="32.0" height="28.0" class="stroke-a" fill="none"/>
<text x="144.0" y="61.0" class="fig-small" text-anchor="middle">3</text>
<rect x="164.0" y="42.0" width="32.0" height="28.0" class="fill-soft-a"/>
<rect x="164.0" y="42.0" width="32.0" height="28.0" class="stroke-a" fill="none"/>
<text x="180.0" y="61.0" class="fig-small" text-anchor="middle">4</text>
<rect x="200.0" y="42.0" width="32.0" height="28.0" class="fill-soft-a"/>
<rect x="200.0" y="42.0" width="32.0" height="28.0" class="stroke-a" fill="none"/>
<text x="216.0" y="61.0" class="fig-small" text-anchor="middle">5</text>
<rect x="236.0" y="42.0" width="32.0" height="28.0" class="fill-soft-a"/>
<rect x="236.0" y="42.0" width="32.0" height="28.0" class="stroke-a" fill="none"/>
<text x="252.0" y="61.0" class="fig-small" text-anchor="middle">6</text>
<rect x="272.0" y="42.0" width="32.0" height="28.0" class="fill-soft-a"/>
<rect x="272.0" y="42.0" width="32.0" height="28.0" class="stroke-a" fill="none"/>
<text x="288.0" y="61.0" class="fig-small" text-anchor="middle">7</text>
<rect x="308.0" y="42.0" width="32.0" height="28.0" class="fill-soft-a"/>
<rect x="308.0" y="42.0" width="32.0" height="28.0" class="stroke-a" fill="none"/>
<text x="324.0" y="61.0" class="fig-small" text-anchor="middle">8</text>
<rect x="344.0" y="42.0" width="32.0" height="28.0" class="fill-soft-a"/>
<rect x="344.0" y="42.0" width="32.0" height="28.0" class="stroke-a" fill="none"/>
<text x="360.0" y="61.0" class="fig-small" text-anchor="middle">9</text>
<rect x="380.0" y="42.0" width="32.0" height="28.0" class="fill-soft-d"/>
<rect x="380.0" y="42.0" width="32.0" height="28.0" class="stroke-d" fill="none"/>
<text x="396.0" y="61.0" class="fig-small fig-bold" text-anchor="middle">8</text>
<rect x="416.0" y="42.0" width="32.0" height="28.0" class="fill-soft-d"/>
<rect x="416.0" y="42.0" width="32.0" height="28.0" class="stroke-d" fill="none"/>
<text x="432.0" y="61.0" class="fig-small fig-bold" text-anchor="middle">7</text>
<text x="20.0" y="92.0" class="fig-small" text-anchor="start">samples 8 and 7 repeat at the end (reflect padding)</text>
<text x="20.0" y="134.0" class="fig-bold" text-anchor="start">folded, period 4: 3 rows by 4 columns</text>
<rect x="40.0" y="148.0" width="58.0" height="32.0" class="fill-soft-a"/>
<rect x="40.0" y="148.0" width="58.0" height="32.0" class="stroke-muted" fill="none"/>
<text x="69.0" y="170.0" class="fig-small" text-anchor="middle">0</text>
<rect x="104.0" y="148.0" width="58.0" height="32.0" class="fill-b"/>
<rect x="104.0" y="148.0" width="58.0" height="32.0" class="stroke-muted" fill="none"/>
<text x="133.0" y="170.0" class="fig-bold" text-anchor="middle">1</text>
<rect x="168.0" y="148.0" width="58.0" height="32.0" class="fill-soft-a"/>
<rect x="168.0" y="148.0" width="58.0" height="32.0" class="stroke-muted" fill="none"/>
<text x="197.0" y="170.0" class="fig-small" text-anchor="middle">2</text>
<rect x="232.0" y="148.0" width="58.0" height="32.0" class="fill-soft-a"/>
<rect x="232.0" y="148.0" width="58.0" height="32.0" class="stroke-muted" fill="none"/>
<text x="261.0" y="170.0" class="fig-small" text-anchor="middle">3</text>
<rect x="40.0" y="186.0" width="58.0" height="32.0" class="fill-soft-a"/>
<rect x="40.0" y="186.0" width="58.0" height="32.0" class="stroke-muted" fill="none"/>
<text x="69.0" y="208.0" class="fig-small" text-anchor="middle">4</text>
<rect x="104.0" y="186.0" width="58.0" height="32.0" class="fill-b"/>
<rect x="104.0" y="186.0" width="58.0" height="32.0" class="stroke-muted" fill="none"/>
<text x="133.0" y="208.0" class="fig-bold" text-anchor="middle">5</text>
<rect x="168.0" y="186.0" width="58.0" height="32.0" class="fill-soft-a"/>
<rect x="168.0" y="186.0" width="58.0" height="32.0" class="stroke-muted" fill="none"/>
<text x="197.0" y="208.0" class="fig-small" text-anchor="middle">6</text>
<rect x="232.0" y="186.0" width="58.0" height="32.0" class="fill-soft-a"/>
<rect x="232.0" y="186.0" width="58.0" height="32.0" class="stroke-muted" fill="none"/>
<text x="261.0" y="208.0" class="fig-small" text-anchor="middle">7</text>
<rect x="40.0" y="224.0" width="58.0" height="32.0" class="fill-soft-a"/>
<rect x="40.0" y="224.0" width="58.0" height="32.0" class="stroke-muted" fill="none"/>
<text x="69.0" y="246.0" class="fig-small" text-anchor="middle">8</text>
<rect x="104.0" y="224.0" width="58.0" height="32.0" class="fill-b"/>
<rect x="104.0" y="224.0" width="58.0" height="32.0" class="stroke-muted" fill="none"/>
<text x="133.0" y="246.0" class="fig-bold" text-anchor="middle">9</text>
<rect x="168.0" y="224.0" width="58.0" height="32.0" class="fill-soft-d"/>
<rect x="168.0" y="224.0" width="58.0" height="32.0" class="stroke-muted" fill="none"/>
<text x="197.0" y="246.0" class="fig-small" text-anchor="middle">8</text>
<rect x="232.0" y="224.0" width="58.0" height="32.0" class="fill-soft-d"/>
<rect x="232.0" y="224.0" width="58.0" height="32.0" class="stroke-muted" fill="none"/>
<text x="261.0" y="246.0" class="fig-small" text-anchor="middle">7</text>
<text x="69.0" y="278.0" class="fig-small" text-anchor="middle">col 0</text>
<text x="133.0" y="278.0" class="fig-small" text-anchor="middle">col 1</text>
<text x="197.0" y="278.0" class="fig-small" text-anchor="middle">col 2</text>
<text x="261.0" y="278.0" class="fig-small" text-anchor="middle">col 3</text>
<text x="318.0" y="180.0" class="fig-small" text-anchor="start">Sample t is in row t // 4, column t % 4.</text>
<text x="318.0" y="202.0" class="fig-small" text-anchor="start">Column 1: samples 1, 5, 9, every 4th.</text>
<text x="318.0" y="224.0" class="fig-small" text-anchor="start">A 1-wide kernel judges one column.</text>
</svg>
Figure 2. Folding with period 4 puts samples one period apart in the same column. Notice that the padded cells in the last row repeat samples 8 and 7.
```

Read it aloud: row $r$ holds samples $rp$ to $rp + p - 1$, so column $c$ holds every $p$-th sample starting at $c$. A kernel 1 wide sees one column at a time, so each column, a signal sampled at $1/p$ of the rate, is judged separately. If $T$ isn't a multiple of $p$, reflection pads the end first:

```python
x = F.pad(x, (0, (-T) % p), mode="reflect")
folded = x.reshape(B, C, -1, p)
```

The periods 2, 3, 5, 7, and 11 are prime so that the sub-discriminators overlap as little as possible: a period of 4 would repeat much of what the period of 2 already sees. HiFi-GAN pairs the MPD with MelGAN's *multi-scale discriminator*, which judges the raw waveform and copies pooled by 2 and 4.

## Give the generator a periodic bias

The generator maps the 80 mel bands to $h_u$ channels, then applies four transposed convolutions that stretch time by 8, 8, 2, and 2, a total of 256, the hop length. After each one, a *multi-receptive-field fusion* block averages residual blocks with kernel sizes 3, 7, and 11, A leaky ReLU is piecewise linear, so a generator built from it assembles every oscillation from many kinks and extrapolates badly to unseen pitches. BigVGAN gives the generator a periodic activation, *Snake*:

$$
f_\alpha(x) = x + \frac{1}{\alpha} \sin^2(\alpha x), \qquad
\frac{\partial f_\alpha}{\partial x} = 1 + \sin(2 \alpha x).
$$

Read it aloud: Snake is the identity plus a periodic bump whose frequency a learned $\alpha$ sets, one per channel. Its slope never goes below 0, and the linear part lets gradients pass as in a residual network. At $x = 0.5$ with $\alpha = 1$, $f = 0.5 + \sin^2(0.5) = 0.7298$. With $\alpha = 5$, $f = 0.5 + 0.2 \sin^2(2.5) = 0.5716$. Figure 3 draws both curves.

```python
a = alpha.reshape(1, -1, 1)
y = x + torch.sin(a * x) ** 2 / a
```

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Snake activation for alpha 1 and alpha 5 against the identity: Snake follows the identity plus a periodic bump">
<line x1="70.0" y1="150.0" x2="510.0" y2="150.0" class="stroke-muted"/>
<line x1="290.0" y1="20.0" x2="290.0" y2="280.0" class="stroke-muted"/>
<line x1="70.0" y1="150.0" x2="70.0" y2="155.0" class="stroke-muted"/>
<text x="70.0" y="170.0" class="fig-small" text-anchor="middle">-3</text>
<line x1="143.3" y1="150.0" x2="143.3" y2="155.0" class="stroke-muted"/>
<text x="143.3" y="170.0" class="fig-small" text-anchor="middle">-2</text>
<line x1="216.7" y1="150.0" x2="216.7" y2="155.0" class="stroke-muted"/>
<text x="216.7" y="170.0" class="fig-small" text-anchor="middle">-1</text>
<line x1="363.3" y1="150.0" x2="363.3" y2="155.0" class="stroke-muted"/>
<text x="363.3" y="170.0" class="fig-small" text-anchor="middle">1</text>
<line x1="436.7" y1="150.0" x2="436.7" y2="155.0" class="stroke-muted"/>
<text x="436.7" y="170.0" class="fig-small" text-anchor="middle">2</text>
<line x1="510.0" y1="150.0" x2="510.0" y2="155.0" class="stroke-muted"/>
<text x="510.0" y="170.0" class="fig-small" text-anchor="middle">3</text>
<line x1="70.0" y1="260.0" x2="510.0" y2="40.0" class="stroke-muted" stroke-dasharray="5 4"/>
<polyline points="70.0,259.3 70.7,258.8 71.5,258.3 72.2,257.8 72.9,257.3 73.7,256.8 74.4,256.3 75.1,255.8 75.9,255.3 76.6,254.8 77.3,254.2 78.1,253.7 78.8,253.1 79.5,252.6 80.3,252.0 81.0,251.5 81.7,250.9 82.5,250.3 83.2,249.7 83.9,249.1 84.7,248.6 85.4,248.0 86.1,247.3 86.9,246.7 87.6,246.1 88.3,245.5 89.1,244.9 89.8,244.2 90.5,243.6 91.3,243.0 92.0,242.3 92.7,241.7 93.5,241.0 94.2,240.3 94.9,239.7 95.7,239.0 96.4,238.3 97.1,237.6 97.9,237.0 98.6,236.3 99.3,235.6 100.1,234.9 100.8,234.2 101.5,233.5 102.3,232.8 103.0,232.1 103.7,231.4 104.5,230.7 105.2,230.0 105.9,229.3 106.7,228.5 107.4,227.8 108.1,227.1 108.9,226.4 109.6,225.6 110.3,224.9 111.1,224.2 111.8,223.5 112.5,222.7 113.3,222.0 114.0,221.3 114.7,220.5 115.5,219.8 116.2,219.1 116.9,218.3 117.7,217.6 118.4,216.9 119.1,216.1 119.9,215.4 120.6,214.7 121.3,213.9 122.1,213.2 122.8,212.5 123.5,211.8 124.3,211.0 125.0,210.3 125.7,209.6 126.5,208.9 127.2,208.1 127.9,207.4 128.7,206.7 129.4,206.0 130.1,205.3 130.9,204.6 131.6,203.9 132.3,203.2 133.1,202.5 133.8,201.8 134.5,201.1 135.3,200.4 136.0,199.7 136.7,199.0 137.5,198.3 138.2,197.6 138.9,197.0 139.7,196.3 140.4,195.6 141.1,195.0 141.9,194.3 142.6,193.7 143.3,193.0 144.1,192.4 144.8,191.7 145.5,191.1 146.3,190.5 147.0,189.9 147.7,189.2 148.5,188.6 149.2,188.0 149.9,187.4 150.7,186.8 151.4,186.2 152.1,185.7 152.9,185.1 153.6,184.5 154.3,184.0 155.1,183.4 155.8,182.8 156.5,182.3 157.3,181.8 158.0,181.2 158.7,180.7 159.5,180.2 160.2,179.7 160.9,179.2 161.7,178.7 162.4,178.2 163.1,177.7 163.9,177.2 164.6,176.7 165.3,176.3 166.1,175.8 166.8,175.4 167.5,174.9 168.3,174.5 169.0,174.1 169.7,173.6 170.5,173.2 171.2,172.8 171.9,172.4 172.7,172.0 173.4,171.6 174.1,171.3 174.9,170.9 175.6,170.5 176.3,170.2 177.1,169.8 177.8,169.5 178.5,169.2 179.3,168.8 180.0,168.5 180.7,168.2 181.5,167.9 182.2,167.6 182.9,167.3 183.7,167.0 184.4,166.8 185.1,166.5 185.9,166.2 186.6,166.0 187.3,165.7 188.1,165.5 188.8,165.3 189.5,165.0 190.3,164.8 191.0,164.6 191.7,164.4 192.5,164.2 193.2,164.0 193.9,163.8 194.7,163.6 195.4,163.4 196.1,163.3 196.9,163.1 197.6,163.0 198.3,162.8 199.1,162.7 199.8,162.5 200.5,162.4 201.3,162.3 202.0,162.1 202.7,162.0 203.5,161.9 204.2,161.8 204.9,161.7 205.7,161.6 206.4,161.5 207.1,161.4 207.9,161.4 208.6,161.3 209.3,161.2 210.1,161.1 210.8,161.1 211.5,161.0 212.3,161.0 213.0,160.9 213.7,160.9 214.5,160.8 215.2,160.8 215.9,160.7 216.7,160.7 217.4,160.7 218.1,160.6 218.9,160.6 219.6,160.6 220.3,160.6 221.1,160.6 221.8,160.5 222.5,160.5 223.3,160.5 224.0,160.5 224.7,160.5 225.5,160.5 226.2,160.5 226.9,160.5 227.7,160.5 228.4,160.5 229.1,160.5 229.9,160.5 230.6,160.5 231.3,160.5 232.1,160.5 232.8,160.5 233.5,160.5 234.3,160.5 235.0,160.5 235.7,160.5 236.5,160.5 237.2,160.5 237.9,160.5 238.7,160.4 239.4,160.4 240.1,160.4 240.9,160.4 241.6,160.4 242.3,160.4 243.1,160.4 243.8,160.4 244.5,160.4 245.3,160.3 246.0,160.3 246.7,160.3 247.5,160.3 248.2,160.2 248.9,160.2 249.7,160.1 250.4,160.1 251.1,160.1 251.9,160.0 252.6,160.0 253.3,159.9 254.1,159.8 254.8,159.8 255.5,159.7 256.3,159.6 257.0,159.6 257.7,159.5 258.5,159.4 259.2,159.3 259.9,159.2 260.7,159.1 261.4,159.0 262.1,158.9 262.9,158.8 263.6,158.6 264.3,158.5 265.1,158.4 265.8,158.2 266.5,158.1 267.3,158.0 268.0,157.8 268.7,157.6 269.5,157.5 270.2,157.3 270.9,157.1 271.7,156.9 272.4,156.7 273.1,156.5 273.9,156.3 274.6,156.1 275.3,155.9 276.1,155.7 276.8,155.4 277.5,155.2 278.3,154.9 279.0,154.7 279.7,154.4 280.5,154.2 281.2,153.9 281.9,153.6 282.7,153.3 283.4,153.0 284.1,152.7 284.9,152.4 285.6,152.1 286.3,151.7 287.1,151.4 287.8,151.1 288.5,150.7 289.3,150.4 290.0,150.0 290.7,149.6 291.5,149.3 292.2,148.9 292.9,148.5 293.7,148.1 294.4,147.7 295.1,147.3 295.9,146.8 296.6,146.4 297.3,146.0 298.1,145.5 298.8,145.1 299.5,144.6 300.3,144.2 301.0,143.7 301.7,143.2 302.5,142.7 303.2,142.2 303.9,141.7 304.7,141.2 305.4,140.7 306.1,140.2 306.9,139.7 307.6,139.1 308.3,138.6 309.1,138.0 309.8,137.5 310.5,136.9 311.3,136.4 312.0,135.8 312.7,135.2 313.5,134.6 314.2,134.0 314.9,133.5 315.7,132.9 316.4,132.2 317.1,131.6 317.9,131.0 318.6,130.4 319.3,129.8 320.1,129.1 320.8,128.5 321.5,127.9 322.3,127.2 323.0,126.6 323.7,125.9 324.5,125.2 325.2,124.6 325.9,123.9 326.7,123.2 327.4,122.6 328.1,121.9 328.9,121.2 329.6,120.5 330.3,119.8 331.1,119.1 331.8,118.4 332.5,117.7 333.3,117.0 334.0,116.3 334.7,115.6 335.5,114.9 336.2,114.2 336.9,113.5 337.7,112.7 338.4,112.0 339.1,111.3 339.9,110.6 340.6,109.8 341.3,109.1 342.1,108.4 342.8,107.7 343.5,106.9 344.3,106.2 345.0,105.5 345.7,104.7 346.5,104.0 347.2,103.3 347.9,102.5 348.7,101.8 349.4,101.1 350.1,100.3 350.9,99.6 351.6,98.9 352.3,98.1 353.1,97.4 353.8,96.7 354.5,96.0 355.3,95.2 356.0,94.5 356.7,93.8 357.5,93.1 358.2,92.3 358.9,91.6 359.7,90.9 360.4,90.2 361.1,89.5 361.9,88.8 362.6,88.1 363.3,87.4 364.1,86.7 364.8,86.0 365.5,85.3 366.3,84.6 367.0,83.9 367.7,83.2 368.5,82.6 369.2,81.9 369.9,81.2 370.7,80.5 371.4,79.9 372.1,79.2 372.9,78.6 373.6,77.9 374.3,77.3 375.1,76.6 375.8,76.0 376.5,75.4 377.3,74.8 378.0,74.1 378.7,73.5 379.5,72.9 380.2,72.3 380.9,71.7 381.7,71.1 382.4,70.6 383.1,70.0 383.9,69.4 384.6,68.8 385.3,68.3 386.1,67.7 386.8,67.2 387.5,66.7 388.3,66.1 389.0,65.6 389.7,65.1 390.5,64.6 391.2,64.1 391.9,63.6 392.7,63.1 393.4,62.6 394.1,62.1 394.9,61.6 395.6,61.2 396.3,60.7 397.1,60.2 397.8,59.8 398.5,59.4 399.3,58.9 400.0,58.5 400.7,58.1 401.5,57.7 402.2,57.3 402.9,56.9 403.7,56.5 404.4,56.1 405.1,55.8 405.9,55.4 406.6,55.0 407.3,54.7 408.1,54.4 408.8,54.0 409.5,53.7 410.3,53.4 411.0,53.1 411.7,52.8 412.5,52.5 413.2,52.2 413.9,51.9 414.7,51.6 415.4,51.3 416.1,51.1 416.9,50.8 417.6,50.6 418.3,50.3 419.1,50.1 419.8,49.9 420.5,49.6 421.3,49.4 422.0,49.2 422.7,49.0 423.5,48.8 424.2,48.6 424.9,48.5 425.7,48.3 426.4,48.1 427.1,48.0 427.9,47.8 428.6,47.6 429.3,47.5 430.1,47.4 430.8,47.2 431.5,47.1 432.3,47.0 433.0,46.9 433.7,46.7 434.5,46.6 435.2,46.5 435.9,46.4 436.7,46.3 437.4,46.3 438.1,46.2 438.9,46.1 439.6,46.0 440.3,46.0 441.1,45.9 441.8,45.8 442.5,45.8 443.3,45.7 444.0,45.7 444.7,45.6 445.5,45.6 446.2,45.6 446.9,45.5 447.7,45.5 448.4,45.5 449.1,45.4 449.9,45.4 450.6,45.4 451.3,45.4 452.1,45.3 452.8,45.3 453.5,45.3 454.3,45.3 455.0,45.3 455.7,45.3 456.5,45.3 457.2,45.3 457.9,45.3 458.7,45.3 459.4,45.3 460.1,45.3 460.9,45.3 461.6,45.3 462.3,45.3 463.1,45.3 463.8,45.3 464.5,45.3 465.3,45.3 466.0,45.3 466.7,45.3 467.5,45.3 468.2,45.3 468.9,45.3 469.7,45.3 470.4,45.2 471.1,45.2 471.9,45.2 472.6,45.2 473.3,45.2 474.1,45.2 474.8,45.2 475.5,45.1 476.3,45.1 477.0,45.1 477.7,45.1 478.5,45.0 479.2,45.0 479.9,45.0 480.7,44.9 481.4,44.9 482.1,44.8 482.9,44.8 483.6,44.7 484.3,44.7 485.1,44.6 485.8,44.5 486.5,44.5 487.3,44.4 488.0,44.3 488.7,44.2 489.5,44.1 490.2,44.0 490.9,43.9 491.7,43.8 492.4,43.7 493.1,43.6 493.9,43.5 494.6,43.4 495.3,43.2 496.1,43.1 496.8,42.9 497.5,42.8 498.3,42.6 499.0,42.5 499.7,42.3 500.5,42.1 501.2,41.9 501.9,41.8 502.7,41.6 503.4,41.4 504.1,41.2 504.9,40.9 505.6,40.7 506.3,40.5 507.1,40.3 507.8,40.0 508.5,39.8 509.3,39.5 510.0,39.3" class="stroke-a" fill="none" stroke-width="2.5"/>
<polyline points="70.0,256.9 70.7,256.2 71.5,255.4 72.2,254.7 72.9,254.0 73.7,253.3 74.4,252.6 75.1,251.9 75.9,251.2 76.6,250.5 77.3,249.9 78.1,249.3 78.8,248.8 79.5,248.2 80.3,247.7 81.0,247.3 81.7,246.8 82.5,246.4 83.2,246.1 83.9,245.8 84.7,245.5 85.4,245.2 86.1,245.0 86.9,244.8 87.6,244.7 88.3,244.5 89.1,244.4 89.8,244.4 90.5,244.3 91.3,244.3 92.0,244.3 92.7,244.3 93.5,244.2 94.2,244.2 94.9,244.2 95.7,244.2 96.4,244.2 97.1,244.2 97.9,244.2 98.6,244.1 99.3,244.0 100.1,243.9 100.8,243.8 101.5,243.7 102.3,243.5 103.0,243.3 103.7,243.0 104.5,242.7 105.2,242.4 105.9,242.0 106.7,241.6 107.4,241.2 108.1,240.7 108.9,240.2 109.6,239.7 110.3,239.1 111.1,238.5 111.8,237.9 112.5,237.3 113.3,236.6 114.0,235.9 114.7,235.2 115.5,234.5 116.2,233.7 116.9,233.0 117.7,232.3 118.4,231.5 119.1,230.8 119.9,230.1 120.6,229.4 121.3,228.7 122.1,228.0 122.8,227.4 123.5,226.8 124.3,226.2 125.0,225.6 125.7,225.1 126.5,224.6 127.2,224.1 127.9,223.7 128.7,223.3 129.4,223.0 130.1,222.7 130.9,222.4 131.6,222.1 132.3,221.9 133.1,221.8 133.8,221.6 134.5,221.5 135.3,221.4 136.0,221.3 136.7,221.3 137.5,221.2 138.2,221.2 138.9,221.2 139.7,221.2 140.4,221.2 141.1,221.2 141.9,221.2 142.6,221.2 143.3,221.2 144.1,221.1 144.8,221.1 145.5,221.0 146.3,220.9 147.0,220.8 147.7,220.6 148.5,220.4 149.2,220.2 149.9,219.9 150.7,219.6 151.4,219.3 152.1,218.9 152.9,218.5 153.6,218.1 154.3,217.6 155.1,217.1 155.8,216.6 156.5,216.0 157.3,215.4 158.0,214.8 158.7,214.1 159.5,213.4 160.2,212.7 160.9,212.0 161.7,211.3 162.4,210.6 163.1,209.8 163.9,209.1 164.6,208.4 165.3,207.7 166.1,206.9 166.8,206.2 167.5,205.6 168.3,204.9 169.0,204.3 169.7,203.6 170.5,203.1 171.2,202.5 171.9,202.0 172.7,201.5 173.4,201.0 174.1,200.6 174.9,200.2 175.6,199.9 176.3,199.6 177.1,199.3 177.8,199.1 178.5,198.9 179.3,198.7 180.0,198.5 180.7,198.4 181.5,198.3 182.2,198.3 182.9,198.2 183.7,198.2 184.4,198.2 185.1,198.2 185.9,198.2 186.6,198.2 187.3,198.2 188.1,198.2 188.8,198.1 189.5,198.1 190.3,198.1 191.0,198.0 191.7,197.9 192.5,197.8 193.2,197.7 193.9,197.5 194.7,197.3 195.4,197.1 196.1,196.8 196.9,196.5 197.6,196.2 198.3,195.8 199.1,195.4 199.8,195.0 200.5,194.5 201.3,194.0 202.0,193.4 202.7,192.8 203.5,192.2 204.2,191.6 204.9,191.0 205.7,190.3 206.4,189.6 207.1,188.9 207.9,188.1 208.6,187.4 209.3,186.7 210.1,186.0 210.8,185.2 211.5,184.5 212.3,183.8 213.0,183.1 213.7,182.4 214.5,181.8 215.2,181.1 215.9,180.5 216.7,179.9 217.4,179.4 218.1,178.9 218.9,178.4 219.6,177.9 220.3,177.5 221.1,177.1 221.8,176.8 222.5,176.5 223.3,176.2 224.0,176.0 224.7,175.8 225.5,175.6 226.2,175.5 226.9,175.4 227.7,175.3 228.4,175.2 229.1,175.2 229.9,175.2 230.6,175.1 231.3,175.1 232.1,175.1 232.8,175.1 233.5,175.1 234.3,175.1 235.0,175.1 235.7,175.1 236.5,175.0 237.2,175.0 237.9,174.9 238.7,174.8 239.4,174.6 240.1,174.5 240.9,174.3 241.6,174.0 242.3,173.7 243.1,173.4 243.8,173.1 244.5,172.7 245.3,172.3 246.0,171.9 246.7,171.4 247.5,170.8 248.2,170.3 248.9,169.7 249.7,169.1 250.4,168.5 251.1,167.8 251.9,167.1 252.6,166.4 253.3,165.7 254.1,165.0 254.8,164.3 255.5,163.5 256.3,162.8 257.0,162.1 257.7,161.3 258.5,160.6 259.2,159.9 259.9,159.3 260.7,158.6 261.4,158.0 262.1,157.4 262.9,156.8 263.6,156.2 264.3,155.7 265.1,155.3 265.8,154.8 266.5,154.4 267.3,154.0 268.0,153.7 268.7,153.4 269.5,153.1 270.2,152.9 270.9,152.7 271.7,152.6 272.4,152.4 273.1,152.3 273.9,152.2 274.6,152.2 275.3,152.1 276.1,152.1 276.8,152.1 277.5,152.1 278.3,152.1 279.0,152.1 279.7,152.1 280.5,152.1 281.2,152.1 281.9,152.0 282.7,152.0 283.4,151.9 284.1,151.8 284.9,151.7 285.6,151.6 286.3,151.4 287.1,151.2 287.8,150.9 288.5,150.7 289.3,150.3 290.0,150.0 290.7,149.6 291.5,149.2 292.2,148.7 292.9,148.2 293.7,147.7 294.4,147.2 295.1,146.6 295.9,146.0 296.6,145.3 297.3,144.6 298.1,144.0 298.8,143.3 299.5,142.5 300.3,141.8 301.0,141.1 301.7,140.4 302.5,139.6 303.2,138.9 303.9,138.2 304.7,137.5 305.4,136.8 306.1,136.1 306.9,135.5 307.6,134.8 308.3,134.2 309.1,133.7 309.8,133.1 310.5,132.6 311.3,132.1 312.0,131.7 312.7,131.3 313.5,130.9 314.2,130.6 314.9,130.3 315.7,130.1 316.4,129.8 317.1,129.7 317.9,129.5 318.6,129.4 319.3,129.3 320.1,129.2 320.8,129.1 321.5,129.1 322.3,129.1 323.0,129.1 323.7,129.1 324.5,129.1 325.2,129.1 325.9,129.1 326.7,129.0 327.4,129.0 328.1,129.0 328.9,128.9 329.6,128.9 330.3,128.8 331.1,128.6 331.8,128.5 332.5,128.3 333.3,128.1 334.0,127.9 334.7,127.6 335.5,127.3 336.2,126.9 336.9,126.5 337.7,126.1 338.4,125.6 339.1,125.1 339.9,124.6 340.6,124.0 341.3,123.4 342.1,122.8 342.8,122.2 343.5,121.5 344.3,120.8 345.0,120.1 345.7,119.4 346.5,118.7 347.2,117.9 347.9,117.2 348.7,116.5 349.4,115.7 350.1,115.0 350.9,114.3 351.6,113.6 352.3,113.0 353.1,112.3 353.8,111.7 354.5,111.1 355.3,110.5 356.0,110.0 356.7,109.5 357.5,109.0 358.2,108.6 358.9,108.2 359.7,107.8 360.4,107.5 361.1,107.2 361.9,107.0 362.6,106.8 363.3,106.6 364.1,106.4 364.8,106.3 365.5,106.2 366.3,106.1 367.0,106.1 367.7,106.1 368.5,106.0 369.2,106.0 369.9,106.0 370.7,106.0 371.4,106.0 372.1,106.0 372.9,106.0 373.6,106.0 374.3,105.9 375.1,105.9 375.8,105.8 376.5,105.7 377.3,105.6 378.0,105.4 378.7,105.2 379.5,105.0 380.2,104.8 380.9,104.5 381.7,104.2 382.4,103.8 383.1,103.4 383.9,103.0 384.6,102.5 385.3,102.0 386.1,101.5 386.8,100.9 387.5,100.3 388.3,99.7 389.0,99.0 389.7,98.3 390.5,97.7 391.2,96.9 391.9,96.2 392.7,95.5 393.4,94.8 394.1,94.0 394.9,93.3 395.6,92.6 396.3,91.9 397.1,91.2 397.8,90.5 398.5,89.8 399.3,89.2 400.0,88.5 400.7,88.0 401.5,87.4 402.2,86.9 402.9,86.4 403.7,85.9 404.4,85.5 405.1,85.1 405.9,84.7 406.6,84.4 407.3,84.2 408.1,83.9 408.8,83.7 409.5,83.5 410.3,83.4 411.0,83.3 411.7,83.2 412.5,83.1 413.2,83.0 413.9,83.0 414.7,83.0 415.4,83.0 416.1,83.0 416.9,83.0 417.6,83.0 418.3,83.0 419.1,83.0 419.8,82.9 420.5,82.9 421.3,82.8 422.0,82.8 422.7,82.7 423.5,82.5 424.2,82.4 424.9,82.2 425.7,81.9 426.4,81.7 427.1,81.4 427.9,81.1 428.6,80.7 429.3,80.3 430.1,79.9 430.8,79.4 431.5,78.9 432.3,78.3 433.0,77.8 433.7,77.1 434.5,76.5 435.2,75.9 435.9,75.2 436.7,74.5 437.4,73.8 438.1,73.1 438.9,72.3 439.6,71.6 440.3,70.9 441.1,70.1 441.8,69.4 442.5,68.7 443.3,68.0 444.0,67.3 444.7,66.7 445.5,66.0 446.2,65.4 446.9,64.8 447.7,64.3 448.4,63.7 449.1,63.3 449.9,62.8 450.6,62.4 451.3,62.0 452.1,61.7 452.8,61.3 453.5,61.1 454.3,60.8 455.0,60.6 455.7,60.5 456.5,60.3 457.2,60.2 457.9,60.1 458.7,60.0 459.4,60.0 460.1,60.0 460.9,60.0 461.6,59.9 462.3,59.9 463.1,59.9 463.8,59.9 464.5,59.9 465.3,59.9 466.0,59.9 466.7,59.8 467.5,59.8 468.2,59.7 468.9,59.6 469.7,59.5 470.4,59.3 471.1,59.1 471.9,58.9 472.6,58.6 473.3,58.3 474.1,58.0 474.8,57.6 475.5,57.2 476.3,56.7 477.0,56.3 477.7,55.7 478.5,55.2 479.2,54.6 479.9,54.0 480.7,53.4 481.4,52.7 482.1,52.0 482.9,51.3 483.6,50.6 484.3,49.9 485.1,49.2 485.8,48.4 486.5,47.7 487.3,47.0 488.0,46.3 488.7,45.6 489.5,44.9 490.2,44.2 490.9,43.5 491.7,42.9 492.4,42.3 493.1,41.7 493.9,41.1 494.6,40.6 495.3,40.1 496.1,39.7 496.8,39.3 497.5,38.9 498.3,38.6 499.0,38.3 499.7,38.0 500.5,37.8 501.2,37.6 501.9,37.4 502.7,37.3 503.4,37.1 504.1,37.1 504.9,37.0 505.6,37.0 506.3,36.9 507.1,36.9 507.8,36.9 508.5,36.9 509.3,36.9 510.0,36.9" class="stroke-b" fill="none" stroke-width="2.5"/>
<circle cx="326.7" cy="123.2" r="5" class="fill-a"/>
<circle cx="326.7" cy="129.0" r="5" class="fill-b"/>
<line x1="90.0" y1="280.0" x2="130.0" y2="280.0" class="stroke-a" stroke-width="2.5"/>
<text x="138.0" y="284.0" class="fig-small" text-anchor="start">alpha = 1: f(0.5) = 0.7298</text>
<line x1="340.0" y1="280.0" x2="380.0" y2="280.0" class="stroke-b" stroke-width="2.5"/>
<text x="388.0" y="284.0" class="fig-small" text-anchor="start">alpha = 5: f(0.5) = 0.5716</text>
<text x="510.0" y="190.0" class="fig-small" text-anchor="end">x</text>
<text x="298.0" y="28.0" class="fig-small" text-anchor="start">f(x); dashed is the identity</text>
</svg>
Figure 3. Snake is the identity plus a bump that repeats. Notice that the curve never decreases, and that a larger alpha makes the bumps faster and smaller.
```

BigVGAN scaled this design to 112 million parameters for unseen speakers, languages, and music. Because Snake creates harmonics above the Nyquist frequency that fold back as aliases, BigVGAN upsamples before each Snake and filters after. The exercise's generator skips that filtering.

> [!KEY]
> A regression vocoder sounds dull because it averages the many waveforms that fit one mel spectrogram. A GAN vocoder swaps that average for a discriminator's judgment, anchors the spectrum with the multi-resolution STFT loss, and generates every sample in one parallel pass.

> [!RECAP]
> - A regression loss averages valid waveforms, and a discriminator asks only whether the audio looks real.
> - LSGAN trains $D$ toward 1 on real and 0 on fake, and $G$ toward 1 on fake.
> - Feature matching sums the mean absolute feature difference over every layer.
> - The MR-STFT loss averages spectral convergence plus log-magnitude L1 over three resolutions.
> - The MPD folds a waveform into $p$ columns, and Snake is $x + \sin^2(\alpha x)/\alpha$.

## Build it step by step

Build the functions in the order the checks test them:

1. **`snake(x, alpha)`** reshapes `alpha` to broadcast over the batch and time, and returns $x + \sin^2(\alpha x)/\alpha$. The common mistake is broadcasting `alpha` along time instead of channels.
2. **`period_reshape(x, period)`** pads the end with `F.pad(..., mode="reflect")` and reshapes to `(B, C, T / p, p)`. The common mistake is a reshape to `(B, C, p, T / p)`, which puts consecutive samples in a column.
3. **`stft_magnitude(x, n_fft, hop, win)`** calls `torch.stft` with a periodic Hann window of length `win`, `center=True`, and reflect padding, and clamps at $10^{-7}$.
4. **`spectral_convergence(mag_true, mag_pred)`** and **`log_magnitude_loss(mag_true, mag_pred)`** are one line each. The common mistake is dividing by the norm of the prediction.
5. **`multi_resolution_stft_loss(x, x_hat, resolutions)`** loops over the resolutions and returns the mean, not the sum.
6. **`lsgan_discriminator_loss`**, **`lsgan_generator_loss`**, and **`feature_matching_loss`** take one `torch.mean` per sub-discriminator, and per layer for feature matching.

## Cost and scaling

Inference is one parallel pass, linear in the audio length. Training is the expensive part: HiFi-GAN ran 2.5 million steps on batches of 16 segments of 8,192 samples, which takes days on one GPU. The exercise's run lasts minutes, so it shows early training.

## Where this breaks

**The mel settings differ.** The mel inside the loss must match the generator's input mel, band for band.

**The discriminator wins.** If $D$ learns much faster than $G$, its scores saturate and the generator gets no gradient. Warm up with the reconstruction losses alone, as the run does, and watch for $D$'s loss collapsing to 0.

**Artifacts at the frame rate.** A transposed convolution whose kernel isn't a multiple of its stride overlaps unevenly, which you hear as a tone at the frame rate. HiFi-GAN uses a kernel of twice the stride. Horizontal lines in the spectrogram give it away.

**The numbers disagree with your ears.** Mel L1 doesn't measure phase, so Griffin-Lim scores a low mel L1 and still sounds metallic, while a GAN vocoder early in training scores worse and can sound more natural. Run a listening test, as [Evaluating synthetic speech](/lessons/050-evaluating-synthetic-speech) explains.

## The exercise

You implement `snake`, `period_reshape`, `stft_magnitude`, `spectral_convergence`, `log_magnitude_loss`, `multi_resolution_stft_loss`, `lsgan_discriminator_loss`, `lsgan_generator_loss`, and `feature_matching_loss`. The checks run on the CPU with synthetic signals. They confirm Snake's hand-worked values and gradient, the reshape for periods 2 to 11 with reflect padding, your STFT against a numpy one, the hand-worked 0.2 and 0.1, and that the MR-STFT loss is zero for a copy, unchanged by gain, and rising with noise. They also check the LSGAN gradient against finite differences, and that a period-5 sub-discriminator built on your reshape changes only one column when one sample changes.

Press **Run** to train. On a GPU, the run trains a V2-sized generator with Snake activations and a five-period MPD on four minutes of LibriSpeech speaker 1272, for at most seven minutes. On a CPU it trains a tiny version for 35 seconds, and its output is still noisy. It then vocodes a held-out clip, prints mel L1 and the MR-STFT loss for your vocoder and Griffin-Lim, and plays the original and both.
