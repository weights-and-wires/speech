---
title: Keyword spotting, endpointing, and language ID
slug: 043-keyword-spotting-and-language-id
part: "Part 4 — Neural speech recognition"
summary: Spot a keyword with smoothing and a hold-off, endpoint with an end-of-query score, and score each in the units a product pays in.
minutes: 40
gpu: false
objectives:
  - Smooth per-frame keyword posteriors with a causal moving average, and fire with a hold-off so one keyword gives one detection.
  - Count false accepts per hour of the whole stream and the false-reject rate, and list the DET points of a threshold sweep.
  - Choose a threshold from a false-accept budget instead of from an equal error rate.
  - Endpoint an utterance with a short timeout when an end-of-query probability is high, and score it by cut-off rate and mean latency.
  - Compute the NIST language-recognition Cavg from trial log-likelihoods, and say what its threshold beta encodes.
prereqs:
  - 020-voice-activity-detection
  - 042-speaker-verification
exercise: 043-keyword-spotting-and-language-id
---

# Keyword spotting, endpointing, and language ID

A smart speaker on a kitchen counter listens all day for one phrase. A detector that is wrong once in 10,000 frames sounds excellent. At 100 frames per second, that is 36 false wake-ups every hour, and the owner stops trusting the speaker.

So accuracy per frame is the wrong measure. The cost of a mistake is paid per hour of use, and a missed keyword is a different mistake from a false wake-up. This chapter builds three small streaming decisions, spotting a keyword, ending an utterance, and naming a language, and scores each in the units a product team uses. Figure 1 shows the first.

```figure
<svg viewBox="0 0 600 316" role="img" aria-label="Keyword score over 3.2 seconds: a keyword, a one-frame spike, and a false bump, with the smoothed score, a threshold of 0.5, and where detections fire with and without a hold-off">
<line x1="60" y1="210.0" x2="570" y2="210.0" class="stroke-grid"/>
<text x="54" y="214.0" class="fig-small" text-anchor="end">0</text>
<line x1="60" y1="125.0" x2="570" y2="125.0" class="stroke-grid"/>
<text x="54" y="129.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="60" y1="40.0" x2="570" y2="40.0" class="stroke-grid"/>
<text x="54" y="44.0" class="fig-small" text-anchor="end">1</text>
<line x1="60" y1="210" x2="570" y2="210" class="stroke-muted"/>
<line x1="60.0" y1="210" x2="60.0" y2="214" class="stroke-muted"/>
<text x="60.0" y="228" class="fig-small" text-anchor="middle">0</text>
<line x1="219.4" y1="210" x2="219.4" y2="214" class="stroke-muted"/>
<text x="219.4" y="228" class="fig-small" text-anchor="middle">1</text>
<line x1="378.8" y1="210" x2="378.8" y2="214" class="stroke-muted"/>
<text x="378.8" y="228" class="fig-small" text-anchor="middle">2</text>
<line x1="538.1" y1="210" x2="538.1" y2="214" class="stroke-muted"/>
<text x="538.1" y="228" class="fig-small" text-anchor="middle">3</text>
<text x="315" y="246" class="fig-small" text-anchor="middle">time (s)</text>
<text x="60" y="26" class="fig-small">keyword score</text>
<polyline points="60.0,199.2 61.6,205.4 63.2,209.3 64.8,209.7 66.4,196.2 68.0,194.5 69.6,199.7 71.2,197.6 72.8,200.8 74.3,194.1 75.9,196.1 77.5,210.0 79.1,195.4 80.7,209.4 82.3,197.6 83.9,207.0 85.5,195.3 87.1,200.8 88.7,204.9 90.3,202.8 91.9,209.5 93.5,207.9 95.1,198.6 96.7,199.0 98.2,199.5 99.8,203.5 101.4,193.0 103.0,193.3 104.6,198.3 106.2,198.9 107.8,45.3 109.4,50.4 111.0,54.7 112.6,44.7 114.2,48.1 115.8,51.7 117.4,48.7 119.0,41.9 120.6,41.1 122.2,50.9 123.8,47.3 125.3,51.5 126.9,46.9 128.5,51.3 130.1,50.3 131.7,41.9 133.3,53.1 134.9,46.4 136.5,55.6 138.1,42.8 139.7,43.6 141.3,52.9 142.9,42.1 144.5,56.0 146.1,51.3 147.7,54.4 149.2,49.3 150.8,43.5 152.4,53.1 154.0,56.1 155.6,50.1 157.2,53.6 158.8,55.5 160.4,47.1 162.0,51.9 163.6,45.6 165.2,53.6 166.8,41.0 168.4,50.8 170.0,55.2 171.6,46.3 173.2,41.2 174.8,49.5 176.3,40.8 177.9,48.5 179.5,49.8 181.1,46.5 182.7,40.1 184.3,40.9 185.9,49.2 187.5,44.1 189.1,48.5 190.7,48.0 192.3,43.6 193.9,50.0 195.5,44.5 197.1,44.9 198.7,41.2 200.2,55.0 201.8,44.6 203.4,194.2 205.0,193.5 206.6,209.7 208.2,195.3 209.8,193.3 211.4,193.7 213.0,207.5 214.6,193.5 216.2,194.9 217.8,196.0 219.4,201.8 221.0,206.0 222.6,196.4 224.2,194.3 225.8,205.5 227.3,200.8 228.9,202.5 230.5,194.2 232.1,209.3 233.7,197.6 235.3,199.6 236.9,209.5 238.5,197.8 240.1,209.7 241.7,197.1 243.3,201.3 244.9,194.2 246.5,208.9 248.1,195.7 249.7,208.9 251.2,204.1 252.8,202.7 254.4,193.6 256.0,200.4 257.6,205.6 259.2,205.9 260.8,194.9 262.4,206.2 264.0,207.9 265.6,205.1 267.2,200.0 268.8,200.6 270.4,196.2 272.0,200.5 273.6,205.1 275.2,203.0 276.8,196.1 278.3,199.3 279.9,193.7 281.5,203.7 283.1,47.6 284.7,199.9 286.3,195.6 287.9,207.5 289.5,203.1 291.1,194.5 292.7,209.3 294.3,196.0 295.9,202.9 297.5,195.9 299.1,209.8 300.7,203.8 302.2,208.7 303.8,198.9 305.4,205.3 307.0,198.1 308.6,194.0 310.2,207.8 311.8,195.3 313.4,209.0 315.0,203.5 316.6,202.7 318.2,201.7 319.8,193.4 321.4,196.8 323.0,204.7 324.6,205.4 326.2,195.3 327.8,195.0 329.3,201.3 330.9,204.1 332.5,193.1 334.1,204.6 335.7,206.9 337.3,195.0 338.9,196.2 340.5,198.6 342.1,193.7 343.7,194.3 345.3,197.3 346.9,195.4 348.5,205.8 350.1,207.6 351.7,198.6 353.2,197.9 354.8,207.2 356.4,203.3 358.0,194.5 359.6,200.5 361.2,200.2 362.8,206.7 364.4,201.1 366.0,201.1 367.6,208.5 369.2,193.3 370.8,200.3 372.4,209.9 374.0,196.9 375.6,193.4 377.2,200.0 378.8,204.6 380.3,206.8 381.9,198.6 383.5,206.7 385.1,200.2 386.7,199.8 388.3,193.6 389.9,208.8 391.5,201.5 393.1,197.4 394.7,207.0 396.3,203.4 397.9,208.9 399.5,197.7 401.1,208.5 402.7,203.3 404.2,195.2 405.8,202.0 407.4,194.5 409.0,197.0 410.6,194.4 412.2,207.8 413.8,208.7 415.4,208.8 417.0,195.2 418.6,199.2 420.2,201.6 421.8,207.2 423.4,198.5 425.0,204.6 426.6,197.9 428.2,202.2 429.8,201.4 431.3,196.6 432.9,208.4 434.5,200.2 436.1,206.6 437.7,196.3 439.3,201.7 440.9,193.2 442.5,206.9 444.1,193.6 445.7,196.4 447.3,201.8 448.9,196.2 450.5,199.8 452.1,198.9 453.7,194.5 455.2,208.9 456.8,195.8 458.4,67.5 460.0,68.5 461.6,57.1 463.2,60.7 464.8,65.7 466.4,66.8 468.0,59.1 469.6,72.5 471.2,62.0 472.8,60.6 474.4,60.4 476.0,68.5 477.6,60.5 479.2,70.2 480.8,67.8 482.3,66.9 483.9,64.8 485.5,72.1 487.1,67.1 488.7,74.0 490.3,61.3 491.9,59.5 493.5,71.6 495.1,62.0 496.7,60.0 498.3,57.3 499.9,59.7 501.5,66.8 503.1,57.3 504.7,57.4 506.2,65.4 507.8,61.2 509.4,58.5 511.0,65.9 512.6,59.3 514.2,62.1 515.8,69.0 517.4,60.9 519.0,64.3 520.6,72.4 522.2,203.3 523.8,208.7 525.4,201.9 527.0,202.7 528.6,202.8 530.2,200.0 531.8,207.9 533.3,194.1 534.9,198.4 536.5,196.0 538.1,194.8 539.7,200.1 541.3,209.3 542.9,197.9 544.5,200.3 546.1,196.0 547.7,201.0 549.3,196.2 550.9,193.1 552.5,204.0 554.1,207.1 555.7,203.3 557.2,197.2 558.8,202.5 560.4,200.0 562.0,207.8 563.6,197.7 565.2,205.2 566.8,206.8 568.4,195.3" class="stroke-muted" fill="none"/>
<polyline points="60.0,199.2 61.6,202.3 63.2,204.6 64.8,205.9 66.4,204.0 68.0,203.0 69.6,201.9 71.2,199.5 72.8,197.7 74.3,197.3 75.9,197.7 77.5,199.7 79.1,199.3 80.7,201.0 82.3,201.7 83.9,203.9 85.5,201.0 87.1,202.0 88.7,201.1 90.3,202.2 91.9,202.7 93.5,205.2 95.1,204.7 96.7,203.6 98.2,202.9 99.8,201.7 101.4,198.7 103.0,197.7 104.6,197.5 106.2,197.4 107.8,165.8 109.4,137.3 111.0,109.5 112.6,78.8 114.2,48.6 115.8,49.9 117.4,49.6 119.0,47.0 120.6,46.3 122.2,46.9 123.8,46.0 125.3,46.5 126.9,47.5 128.5,49.6 130.1,49.5 131.7,48.4 133.3,48.7 134.9,48.6 136.5,49.5 138.1,48.0 139.7,48.3 141.3,48.3 142.9,47.4 144.5,47.5 146.1,49.2 147.7,51.4 149.2,50.6 150.8,50.9 152.4,50.3 154.0,51.3 155.6,50.4 157.2,51.3 158.8,53.7 160.4,52.5 162.0,51.7 163.6,50.7 165.2,50.7 166.8,47.8 168.4,48.6 170.0,49.2 171.6,49.4 173.2,46.9 174.8,48.6 176.3,46.6 177.9,45.3 179.5,46.0 181.1,47.0 182.7,45.1 184.3,45.1 185.9,45.3 187.5,44.1 189.1,44.6 190.7,46.1 192.3,46.7 193.9,46.9 195.5,46.9 197.1,46.2 198.7,44.8 200.2,47.1 201.8,46.0 203.4,76.0 205.0,105.7 206.6,139.4 208.2,167.5 209.8,197.2 211.4,197.1 213.0,199.9 214.6,196.7 216.2,196.6 217.8,197.1 219.4,198.7 221.0,198.4 222.6,199.0 224.2,198.9 225.8,200.8 227.3,200.6 228.9,199.9 230.5,199.5 232.1,202.5 233.7,200.9 235.3,200.6 236.9,202.0 238.5,202.7 240.1,202.8 241.7,202.7 243.3,203.1 244.9,200.0 246.5,202.2 248.1,199.4 249.7,201.8 251.2,202.4 252.8,204.1 254.4,201.0 256.0,201.9 257.6,201.3 259.2,201.6 260.8,200.1 262.4,202.6 264.0,204.1 265.6,204.0 267.2,202.8 268.8,204.0 270.4,202.0 272.0,200.5 273.6,200.5 275.2,201.1 276.8,200.2 278.3,200.8 279.9,199.4 281.5,199.2 283.1,168.1 284.7,168.9 286.3,168.1 287.9,170.9 289.5,170.7 291.1,200.1 292.7,202.0 294.3,202.1 295.9,201.2 297.5,199.7 299.1,202.8 300.7,201.7 302.2,204.2 303.8,203.4 305.4,205.3 307.0,203.0 308.6,201.0 310.2,200.8 311.8,200.1 313.4,200.8 315.0,201.9 316.6,203.7 318.2,202.4 319.8,202.1 321.4,199.6 323.0,199.9 324.6,200.4 326.2,199.1 327.8,199.5 329.3,200.4 330.9,200.2 332.5,197.8 334.1,199.6 335.7,202.0 337.3,200.8 338.9,199.2 340.5,200.3 342.1,198.1 343.7,195.6 345.3,196.0 346.9,195.9 348.5,197.3 350.1,200.1 351.7,200.9 353.2,201.0 354.8,203.4 356.4,202.9 358.0,200.3 359.6,200.7 361.2,201.1 362.8,201.0 364.4,200.6 366.0,201.9 367.6,203.5 369.2,202.1 370.8,200.8 372.4,202.6 374.0,201.8 375.6,198.7 377.2,200.1 378.8,200.9 380.3,200.3 381.9,200.7 383.5,203.3 385.1,203.4 386.7,202.4 388.3,199.8 389.9,201.8 391.5,200.8 393.1,200.2 394.7,201.6 396.3,203.6 397.9,203.6 399.5,202.9 401.1,205.1 402.7,204.4 404.2,202.7 405.8,201.3 407.4,200.7 409.0,198.4 410.6,196.6 412.2,199.1 413.8,200.5 415.4,203.4 417.0,203.0 418.6,204.0 420.2,202.7 421.8,202.4 423.4,200.4 425.0,202.2 426.6,202.0 428.2,202.1 429.8,200.9 431.3,200.5 432.9,201.3 434.5,201.7 436.1,202.6 437.7,201.6 439.3,202.6 440.9,199.6 442.5,200.9 444.1,198.3 445.7,198.4 447.3,198.4 448.9,199.0 450.5,197.6 452.1,198.6 453.7,198.2 455.2,199.6 456.8,199.6 458.4,173.1 460.0,147.0 461.6,119.6 463.2,89.9 464.8,63.9 466.4,63.8 468.0,61.9 469.6,65.0 471.2,65.2 472.8,64.2 474.4,62.9 476.0,64.8 477.6,62.4 479.2,64.0 480.8,65.5 482.3,66.8 483.9,66.0 485.5,68.4 487.1,67.7 488.7,69.0 490.3,67.9 491.9,66.8 493.5,66.7 495.1,65.7 496.7,62.9 498.3,62.1 499.9,62.1 501.5,61.2 503.1,60.2 504.7,59.7 506.2,61.3 507.8,61.6 509.4,60.0 511.0,61.7 512.6,62.1 514.2,61.4 515.8,63.0 517.4,63.4 519.0,63.1 520.6,65.7 522.2,94.0 523.8,121.9 525.4,150.1 527.0,177.8 528.6,203.9 530.2,203.2 531.8,203.1 533.3,201.5 534.9,200.6 536.5,199.3 538.1,198.2 539.7,196.7 541.3,199.7 542.9,199.6 544.5,200.5 546.1,200.7 547.7,200.9 549.3,198.3 550.9,197.3 552.5,198.0 554.1,200.3 555.7,200.7 557.2,200.9 558.8,202.8 560.4,202.0 562.0,202.2 563.6,201.0 565.2,202.7 566.8,203.5 568.4,202.6" class="stroke-b" fill="none"/>
<line x1="60" y1="125.0" x2="570" y2="125.0" class="stroke-ink dashed"/>
<text x="570" y="120.0" class="fig-small" text-anchor="end">threshold 0.5</text>
<text x="155.6" y="34" class="fig-small" text-anchor="middle">keyword</text>
<text x="283.1" y="34" class="fig-small" text-anchor="middle">spike</text>
<text x="490.3" y="34" class="fig-small" text-anchor="middle">false bump</text>
<circle cx="111.0" cy="270" r="2.5" class="fill-bad"/>
<circle cx="112.6" cy="270" r="2.5" class="fill-bad"/>
<circle cx="114.2" cy="270" r="2.5" class="fill-bad"/>
<circle cx="115.8" cy="270" r="2.5" class="fill-bad"/>
<circle cx="117.4" cy="270" r="2.5" class="fill-bad"/>
<circle cx="119.0" cy="270" r="2.5" class="fill-bad"/>
<circle cx="120.6" cy="270" r="2.5" class="fill-bad"/>
<circle cx="122.2" cy="270" r="2.5" class="fill-bad"/>
<circle cx="123.8" cy="270" r="2.5" class="fill-bad"/>
<circle cx="125.3" cy="270" r="2.5" class="fill-bad"/>
<circle cx="126.9" cy="270" r="2.5" class="fill-bad"/>
<circle cx="128.5" cy="270" r="2.5" class="fill-bad"/>
<circle cx="130.1" cy="270" r="2.5" class="fill-bad"/>
<circle cx="131.7" cy="270" r="2.5" class="fill-bad"/>
<circle cx="133.3" cy="270" r="2.5" class="fill-bad"/>
<circle cx="134.9" cy="270" r="2.5" class="fill-bad"/>
<circle cx="136.5" cy="270" r="2.5" class="fill-bad"/>
<circle cx="138.1" cy="270" r="2.5" class="fill-bad"/>
<circle cx="139.7" cy="270" r="2.5" class="fill-bad"/>
<circle cx="141.3" cy="270" r="2.5" class="fill-bad"/>
<circle cx="142.9" cy="270" r="2.5" class="fill-bad"/>
<circle cx="144.5" cy="270" r="2.5" class="fill-bad"/>
<circle cx="146.1" cy="270" r="2.5" class="fill-bad"/>
<circle cx="147.7" cy="270" r="2.5" class="fill-bad"/>
<circle cx="149.2" cy="270" r="2.5" class="fill-bad"/>
<circle cx="150.8" cy="270" r="2.5" class="fill-bad"/>
<circle cx="152.4" cy="270" r="2.5" class="fill-bad"/>
<circle cx="154.0" cy="270" r="2.5" class="fill-bad"/>
<circle cx="155.6" cy="270" r="2.5" class="fill-bad"/>
<circle cx="157.2" cy="270" r="2.5" class="fill-bad"/>
<circle cx="158.8" cy="270" r="2.5" class="fill-bad"/>
<circle cx="160.4" cy="270" r="2.5" class="fill-bad"/>
<circle cx="162.0" cy="270" r="2.5" class="fill-bad"/>
<circle cx="163.6" cy="270" r="2.5" class="fill-bad"/>
<circle cx="165.2" cy="270" r="2.5" class="fill-bad"/>
<circle cx="166.8" cy="270" r="2.5" class="fill-bad"/>
<circle cx="168.4" cy="270" r="2.5" class="fill-bad"/>
<circle cx="170.0" cy="270" r="2.5" class="fill-bad"/>
<circle cx="171.6" cy="270" r="2.5" class="fill-bad"/>
<circle cx="173.2" cy="270" r="2.5" class="fill-bad"/>
<circle cx="174.8" cy="270" r="2.5" class="fill-bad"/>
<circle cx="176.3" cy="270" r="2.5" class="fill-bad"/>
<circle cx="177.9" cy="270" r="2.5" class="fill-bad"/>
<circle cx="179.5" cy="270" r="2.5" class="fill-bad"/>
<circle cx="181.1" cy="270" r="2.5" class="fill-bad"/>
<circle cx="182.7" cy="270" r="2.5" class="fill-bad"/>
<circle cx="184.3" cy="270" r="2.5" class="fill-bad"/>
<circle cx="185.9" cy="270" r="2.5" class="fill-bad"/>
<circle cx="187.5" cy="270" r="2.5" class="fill-bad"/>
<circle cx="189.1" cy="270" r="2.5" class="fill-bad"/>
<circle cx="190.7" cy="270" r="2.5" class="fill-bad"/>
<circle cx="192.3" cy="270" r="2.5" class="fill-bad"/>
<circle cx="193.9" cy="270" r="2.5" class="fill-bad"/>
<circle cx="195.5" cy="270" r="2.5" class="fill-bad"/>
<circle cx="197.1" cy="270" r="2.5" class="fill-bad"/>
<circle cx="198.7" cy="270" r="2.5" class="fill-bad"/>
<circle cx="200.2" cy="270" r="2.5" class="fill-bad"/>
<circle cx="201.8" cy="270" r="2.5" class="fill-bad"/>
<circle cx="203.4" cy="270" r="2.5" class="fill-bad"/>
<circle cx="205.0" cy="270" r="2.5" class="fill-bad"/>
<circle cx="461.6" cy="270" r="2.5" class="fill-bad"/>
<circle cx="463.2" cy="270" r="2.5" class="fill-bad"/>
<circle cx="464.8" cy="270" r="2.5" class="fill-bad"/>
<circle cx="466.4" cy="270" r="2.5" class="fill-bad"/>
<circle cx="468.0" cy="270" r="2.5" class="fill-bad"/>
<circle cx="469.6" cy="270" r="2.5" class="fill-bad"/>
<circle cx="471.2" cy="270" r="2.5" class="fill-bad"/>
<circle cx="472.8" cy="270" r="2.5" class="fill-bad"/>
<circle cx="474.4" cy="270" r="2.5" class="fill-bad"/>
<circle cx="476.0" cy="270" r="2.5" class="fill-bad"/>
<circle cx="477.6" cy="270" r="2.5" class="fill-bad"/>
<circle cx="479.2" cy="270" r="2.5" class="fill-bad"/>
<circle cx="480.8" cy="270" r="2.5" class="fill-bad"/>
<circle cx="482.3" cy="270" r="2.5" class="fill-bad"/>
<circle cx="483.9" cy="270" r="2.5" class="fill-bad"/>
<circle cx="485.5" cy="270" r="2.5" class="fill-bad"/>
<circle cx="487.1" cy="270" r="2.5" class="fill-bad"/>
<circle cx="488.7" cy="270" r="2.5" class="fill-bad"/>
<circle cx="490.3" cy="270" r="2.5" class="fill-bad"/>
<circle cx="491.9" cy="270" r="2.5" class="fill-bad"/>
<circle cx="493.5" cy="270" r="2.5" class="fill-bad"/>
<circle cx="495.1" cy="270" r="2.5" class="fill-bad"/>
<circle cx="496.7" cy="270" r="2.5" class="fill-bad"/>
<circle cx="498.3" cy="270" r="2.5" class="fill-bad"/>
<circle cx="499.9" cy="270" r="2.5" class="fill-bad"/>
<circle cx="501.5" cy="270" r="2.5" class="fill-bad"/>
<circle cx="503.1" cy="270" r="2.5" class="fill-bad"/>
<circle cx="504.7" cy="270" r="2.5" class="fill-bad"/>
<circle cx="506.2" cy="270" r="2.5" class="fill-bad"/>
<circle cx="507.8" cy="270" r="2.5" class="fill-bad"/>
<circle cx="509.4" cy="270" r="2.5" class="fill-bad"/>
<circle cx="511.0" cy="270" r="2.5" class="fill-bad"/>
<circle cx="512.6" cy="270" r="2.5" class="fill-bad"/>
<circle cx="514.2" cy="270" r="2.5" class="fill-bad"/>
<circle cx="515.8" cy="270" r="2.5" class="fill-bad"/>
<circle cx="517.4" cy="270" r="2.5" class="fill-bad"/>
<circle cx="519.0" cy="270" r="2.5" class="fill-bad"/>
<circle cx="520.6" cy="270" r="2.5" class="fill-bad"/>
<circle cx="522.2" cy="270" r="2.5" class="fill-bad"/>
<circle cx="523.8" cy="270" r="2.5" class="fill-bad"/>
<text x="300" y="274" class="fig-small">no hold-off: 100 detections</text>
<circle cx="111.0" cy="292" r="2.5" class="fill-b"/>
<circle cx="461.6" cy="292" r="2.5" class="fill-b"/>
<text x="300" y="296" class="fig-small">hold-off 100: 2 detections</text>
</svg>
Figure 1. The raw score (gray) fires on a one-frame spike. The smoothed score (orange) does not, and a hold-off turns its 100 detections of the keyword and the bump into two.
```

## What you already know

- A recognizer scores each frame, and a frame at a 10 ms hop is 0.01 s.
- [Voice activity detection](/lessons/020-voice-activity-detection) ends an utterance when non-speech frames after speech exceed a timeout.
- [Speaker verification](/lessons/042-speaker-verification) trades misses against false accepts, and reports EER and minDCF.

## Spot a keyword in a stream

A *keyword spotter* is a small network that reads a short window of features and outputs, once per frame, the probability that the keyword has ended. Apple's "Hey Siri" post describes a two-pass design, where a small first network wakes a larger one to confirm. Warden's Speech Commands dataset (2018) is the usual open benchmark.

The obvious detector fires when the score reaches a threshold. It fails twice. A single noisy frame can cross the threshold, and a real keyword keeps the score high for many frames, so the detector fires on each.

Chen, Parada, and Heigold (2014) fix the first failure by smoothing the posteriors over a sliding window. For a window of $w$ frames, the causal moving average is

$$
\bar p_t = \frac{1}{t - s + 1} \sum_{u=s}^{t} p_u, \qquad s = \max(0,\, t - w + 1).
$$

Read it aloud: average the current frame and the $w - 1$ before it, using fewer near the start. *Causal* means it never reads a future frame.

Take scores $[0.1, 0.2, 0.9, 0.1, 0.8, 0.9, 0.7, 0.2, 0.1, 0.1, 0.1]$ and $w = 3$. The smoothed values are $[0.1, 0.15, 0.4, 0.4, 0.6, 0.6, 0.8, 0.6, 0.33, 0.13, 0.1]$. At a threshold of 0.5, the raw score fires at frames 2, 4, 5, and 6, and the smoothed score at frames 4 to 7. The lone 0.9 at frame 2 no longer counts, at the price of a delay.

> [!KEY]
> Smoothing trades latency for fewer false accepts: a spike must stay high for most of a window before it counts.

Smoothing does not fix the second failure. In Figure 1 the smoothed keyword stays above 0.5 for about 60 frames. A *hold-off*, also called a refractory period, fixes it: after a detection at frame $t$, ignore frames $t + 1$ to $t + h$.

Run it by hand. The scores are $[0.1, 0.6, 0.7, 0.8, 0.2, 0.9, 0.9, 0.1, 0.9, 0.9]$, the threshold is 0.5, and the hold-off is 3.

1. Frame 1 is the first score at or above 0.5, so it fires. Frames 2 to 4 are suppressed.
2. Frame 5 is above the threshold and past frame 4, so it fires. Frames 6 to 8 are suppressed, which includes frame 8.
3. Frame 9 is past frame 8, so it fires.

The detections are $[1, 5, 9]$. With a hold-off of 0, seven frames fire.

## Score a detector in deployment units

A detection inside a keyword's span is correct, and one outside every span is a *false accept*. A keyword with no detection is a *false reject*. Two numbers summarize a stream:

$$
\text{FA/hour} = \frac{\text{detections outside every keyword}}{\text{hours of the whole stream}}, \qquad
\text{FRR} = \frac{\text{keywords with no detection}}{\text{keywords}}.
$$

The denominator is the whole stream, the time the device spends listening.

The exercise plants a six-minute stream, 0.1 hour: 24 keywords, four of them weak (score near 0.35), six false bumps, and three one-frame spikes. Table 1 runs `smooth_posteriors`, `detect`, and `evaluate_stream` at a threshold of 0.5.

| Smoothing window | Hold-off (frames) | Detections | False accepts per hour | FRR |
|---|---|---|---|---|
| 1 (none) | 0 | 1443 | 2430 | 0.167 |
| 1 (none) | 100 | 29 | 90 | 0.167 |
| 5 | 0 | 1440 | 2800 | 0.167 |
| 5 | 100 | 26 | 60 | 0.167 |

Table 1. Six false accepts in 0.1 hour is 60 per hour, and the FRR is the four weak keywords out of 24. Smoothing alone is the worst row, because a smoothed bump is wider and fires on more frames.

> [!WARNING]
> Dividing by keyword time instead of stream time makes a rate grow as keywords get rarer, which is backwards. Check the denominator before you trust a rate.

Every threshold gives one pair of rates, and a *DET curve* lists those pairs. [Speaker verification](/lessons/042-speaker-verification) defines EER and minDCF on this curve. With targets $[0.9, 0.7, 0.5, 0.3]$ and non-targets $[0.6, 0.4, 0.2, 0.1]$, accepting a score at or above the threshold gives these points. The exercise tries every distinct score, then $+\infty$.

| Threshold | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | $+\infty$ |
|---|---|---|---|---|---|---|
| $P_\text{miss}$ | 0 | 0.25 | 0.25 | 0.5 | 0.5 | 1 |
| $P_\text{fa}$ | 0.5 | 0.5 | 0.25 | 0.25 | 0 | 0 |

A wake-word detector does not sit at the equal-error point. It starts from a budget, such as one false wake-up per day, and takes the lowest threshold that meets it. In the toy set, a budget of $P_\text{fa} \le 0.25$ gives a threshold of 0.5 and a miss rate of 0.25.

> [!KEY]
> A wake-word detector is judged at the one DET point that meets a false-accepts-per-hour budget, not at the equal error rate.

## Decide that the query is over

[Voice activity detection](/lessons/020-voice-activity-detection) ends an utterance after a fixed timeout of silence. One number cannot serve everyone. A short timeout answers fast and cuts off a person who pauses to think. A long one waits through every pause, so each turn feels slow.

A model that reads the words so far can say how likely the query is complete: "set a timer for five" is probably not, and "set a timer for five minutes" is. Call that per-frame probability `eoq`, for end of query. The endpointer uses two timeouts. At a non-speech frame with `eoq >= threshold` the limit is `short_timeout`, and otherwise it is `long_timeout`. The endpoint is the first frame where the count of consecutive non-speech frames exceeds the limit.

Figure 2 runs three policies on one utterance at a 100 ms hop. Speech covers frames 0 to 5 and 9 to 11, with a three-frame pause between.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="One utterance in 18 frames of 100 ms: speech, a pause, more speech, then silence, with the end-of-query probability and the frame where each of three endpointers fires">
<text x="12" y="44" class="fig-small">frame</text>
<text x="103" y="44" class="fig-small" text-anchor="middle">0</text>
<rect x="90" y="54" width="26" height="26" rx="3" class="fill-soft-a stroke-a"/>
<rect x="90" y="148.0" width="26" height="0.0" class="fill-soft-b stroke-b"/>
<rect x="118" y="54" width="26" height="26" rx="3" class="fill-soft-a stroke-a"/>
<rect x="118" y="148.0" width="26" height="0.0" class="fill-soft-b stroke-b"/>
<text x="159" y="44" class="fig-small" text-anchor="middle">2</text>
<rect x="146" y="54" width="26" height="26" rx="3" class="fill-soft-a stroke-a"/>
<rect x="146" y="148.0" width="26" height="0.0" class="fill-soft-b stroke-b"/>
<rect x="174" y="54" width="26" height="26" rx="3" class="fill-soft-a stroke-a"/>
<rect x="174" y="148.0" width="26" height="0.0" class="fill-soft-b stroke-b"/>
<text x="215" y="44" class="fig-small" text-anchor="middle">4</text>
<rect x="202" y="54" width="26" height="26" rx="3" class="fill-soft-a stroke-a"/>
<rect x="202" y="148.0" width="26" height="0.0" class="fill-soft-b stroke-b"/>
<rect x="230" y="54" width="26" height="26" rx="3" class="fill-soft-a stroke-a"/>
<rect x="230" y="148.0" width="26" height="0.0" class="fill-soft-b stroke-b"/>
<text x="271" y="44" class="fig-small" text-anchor="middle">6</text>
<rect x="258" y="54" width="26" height="26" rx="3" class="fill-sunk stroke-muted"/>
<rect x="258" y="142.0" width="26" height="6.0" class="fill-soft-b stroke-b"/>
<rect x="286" y="54" width="26" height="26" rx="3" class="fill-sunk stroke-muted"/>
<rect x="286" y="142.0" width="26" height="6.0" class="fill-soft-b stroke-b"/>
<text x="327" y="44" class="fig-small" text-anchor="middle">8</text>
<rect x="314" y="54" width="26" height="26" rx="3" class="fill-sunk stroke-muted"/>
<rect x="314" y="136.0" width="26" height="12.0" class="fill-soft-b stroke-b"/>
<rect x="342" y="54" width="26" height="26" rx="3" class="fill-soft-a stroke-a"/>
<rect x="342" y="148.0" width="26" height="0.0" class="fill-soft-b stroke-b"/>
<text x="383" y="44" class="fig-small" text-anchor="middle">10</text>
<rect x="370" y="54" width="26" height="26" rx="3" class="fill-soft-a stroke-a"/>
<rect x="370" y="148.0" width="26" height="0.0" class="fill-soft-b stroke-b"/>
<rect x="398" y="54" width="26" height="26" rx="3" class="fill-soft-a stroke-a"/>
<rect x="398" y="148.0" width="26" height="0.0" class="fill-soft-b stroke-b"/>
<text x="439" y="44" class="fig-small" text-anchor="middle">12</text>
<rect x="426" y="54" width="26" height="26" rx="3" class="fill-sunk stroke-muted"/>
<rect x="426" y="130.0" width="26" height="18.0" class="fill-soft-b stroke-b"/>
<rect x="454" y="54" width="26" height="26" rx="3" class="fill-sunk stroke-muted"/>
<rect x="454" y="106.0" width="26" height="42.0" class="fill-soft-b stroke-b"/>
<text x="495" y="44" class="fig-small" text-anchor="middle">14</text>
<rect x="482" y="54" width="26" height="26" rx="3" class="fill-sunk stroke-muted"/>
<rect x="482" y="94.0" width="26" height="54.0" class="fill-soft-b stroke-b"/>
<rect x="510" y="54" width="26" height="26" rx="3" class="fill-sunk stroke-muted"/>
<rect x="510" y="94.0" width="26" height="54.0" class="fill-soft-b stroke-b"/>
<text x="551" y="44" class="fig-small" text-anchor="middle">16</text>
<rect x="538" y="54" width="26" height="26" rx="3" class="fill-sunk stroke-muted"/>
<rect x="538" y="94.0" width="26" height="54.0" class="fill-soft-b stroke-b"/>
<rect x="566" y="54" width="26" height="26" rx="3" class="fill-sunk stroke-muted"/>
<rect x="566" y="94.0" width="26" height="54.0" class="fill-soft-b stroke-b"/>
<text x="12" y="72" class="fig-small">speech</text>
<text x="12" y="120" class="fig-small">end of</text><text x="12" y="134" class="fig-small">query</text>
<line x1="90" y1="118.0" x2="594" y2="118.0" class="stroke-ink dashed"/>
<text x="84" y="122.0" class="fig-small" text-anchor="end">0.5</text>
<circle cx="327" cy="182" r="6" class="fill-bad"/>
<text x="315" y="186" class="fig-small" text-anchor="end">fixed 2 frames: frame 8</text>
<circle cx="467" cy="208" r="6" class="fill-b"/>
<text x="455" y="212" class="fig-small" text-anchor="end">end-of-query, 1 and 5: frame 13</text>
<circle cx="579" cy="234" r="6" class="fill-c"/>
<text x="567" y="238" class="fig-small" text-anchor="end">fixed 5 frames: frame 17</text>
<line x1="424" y1="50" x2="424" y2="244" class="stroke-muted dashed"/>
</svg>
Figure 2. A fixed 2-frame timeout fires inside the pause, at frame 8, and cuts the speaker off. A fixed 5-frame timeout waits until frame 17. The end-of-query endpointer fires at frame 13, once the probability rises.
```

1. With a fixed timeout of 2, the pause at frames 6, 7, and 8 counts 1, 2, 3, so the endpoint is 8 while the user is still talking.
2. With a fixed timeout of 5, the pause survives, and the count reaches 6 at frame 17. The latency is 17 - 11 = 6 frames, or 600 ms.
3. With short 1 and long 5, the pause frames have `eoq` of 0.1 to 0.2, so the limit stays 5. At frame 13 the count is 2 and `eoq` is 0.7, so the limit is 1 and the endpoint is 13, a latency of 200 ms.

Two numbers score an endpointer over many utterances. The *cut-off rate* is the share of utterances whose endpoint comes before the last speech frame. The *mean latency* is the average gap between the last speech frame and the endpoint, over the utterances not cut off. A cut-off loses the query, so a team fixes it first. Table 2 adds utterance B, with no pause, and C, with a six-frame hesitation.

| Policy | Endpoints (A, B, C) | Cut-offs | Mean latency |
|---|---|---|---|
| Fixed 2 frames | 8, 10, 6 | 2 of 3 | 300 ms |
| Fixed 5 frames | 17, 13, 9 | 1 of 3 | 600 ms |
| Short 1, long 8, with `eoq` | 13, 11, 14 | 0 of 3 | 267 ms |

Table 2. The end-of-query endpointer has no cut-offs and the lowest latency, because it spends its long timeout only when the query looks unfinished.

## Name the language

Language identification (LID) classifies a whole utterance. A network emits per-frame log-probabilities over $N$ languages, and you average them over frames into one score $\ell_j$ per language. [Whisper](/lessons/041-whisper) does the same job with one softmax over its language tokens.

The NIST language-recognition evaluations score LID as $N$ detection problems. For one trial, the log-likelihood ratio of language $i$ against the rest is

$$
\text{LLR}_i = \ell_i - \log\!\Big(\frac{1}{N-1} \sum_{j \ne i} e^{\ell_j}\Big).
$$

It says how much better language $i$ explains the trial than the average of the others. Language $i$ is accepted when $\text{LLR}_i \ge \log \beta$, where $\beta = C_\text{fa}(1 - P_\text{tar}) / (C_\text{miss}\, P_\text{tar})$ weights the two errors as in the minDCF of speaker verification. The NIST 2017 plan uses costs of 1 and $P_\text{tar}$ of 0.5 ($\beta = 1$) and 0.1 ($\beta = 9$). The average cost is

$$
C_\text{avg}(\beta) = \frac{1}{N}\Big[\sum_{T} P_\text{miss}(T) + \frac{\beta}{N-1} \sum_{T} \sum_{L \ne T} P_\text{fa}(T, L)\Big].
$$

$P_\text{miss}(T)$ is the share of language-$T$ trials that reject $T$, and $P_\text{fa}(T, L)$ is the share of language-$L$ trials that accept $T$. A perfect system scores 0.

Table 3 has three languages and six trials, two per language.

| Trial | True language | $\ell$ | LLRs | Accepted at $\beta = 1$ |
|---|---|---|---|---|
| 1 | 0 | 2, 0, 0 | 2.00, -1.43, -1.43 | 0 |
| 2 | 0 | 0, 1, 0 | -0.62, 1.00, -0.62 | 1 |
| 3 | 1 | 0, 2, 0 | -1.43, 2.00, -1.43 | 1 |
| 4 | 1 | 1, 1, 0 | 0.38, 0.38, -1.00 | 0 and 1 |
| 5 | 2 | 0, 0, 3 | -2.36, -2.36, 3.00 | 2 |
| 6 | 2 | 0, 2, 0 | -1.43, 2.00, -1.43 | 1 |

Table 3. Trial 2 rejects its true language, and trial 4 accepts a wrong one.

At $\beta = 1$ the miss rates are $1/2$, $0$, and $1/2$. Three false-accept terms equal $1/2$: language 0 on trial 4, and language 1 on trials 2 and 6. They sum to 1.5, so $C_\text{avg} = (1 + 1.5/2)/3 = 0.583$. At $\beta = 9$ the threshold is $\log 9 \approx 2.20$. Only trial 5 clears it, so the misses are 1, 1, and $1/2$, there are no false accepts, and $C_\text{avg} = 2.5/3 = 0.833$. A larger $\beta$ punishes false accepts more, so the system must be surer.

> [!KEY]
> Wake word, endpoint, and language ID share one habit: pick the error you can afford, state it in the unit the product pays in, and report the other error at that point.

> [!RECAP]
> - Smooth the posteriors with a causal moving average, then fire with a hold-off.
> - Report false accepts per hour of the whole stream and the FRR, at a threshold that meets a budget.
> - An endpointer scores by cut-off rate and mean latency, and an end-of-query probability buys a short timeout.
> - $C_\text{avg}$ averages misses and prior-weighted false accepts over languages, with the threshold $\log \beta$ on each LLR.

## Build it step by step

The checks test the functions in this order.

1. **`smooth_posteriors(posteriors, window)`** uses a cumulative sum with a zero in front. The usual mistake is dividing by `window` at the start.
2. **`detect(scores, threshold, holdoff)`** loops over the frames at or above the threshold. The usual mistake is allowing frame $t + h$, or no hold-off.
3. **`evaluate_stream(detections, spans, n_frames, hop_seconds)`** returns FA per hour and FRR. A span includes its start and excludes its end. The usual mistake is dividing by the keyword duration.
4. **`det_points(target_scores, nontarget_scores)`** returns thresholds, miss rates, and false-accept rates. Use `<` for a miss, `>=` for a false accept, and end with `np.inf`.
5. **`endpoint(vad, eoq, short_timeout, long_timeout, threshold)`** loops once with a started flag and a quiet count. It returns -1 if it never fires.
6. **`endpoint_metrics(decisions, last_speech, hop_ms)`** returns the cut-off rate and the mean latency in milliseconds. Cut-offs stay out of the latency mean.
7. **`language_llr(loglik)`** uses `logsumexp` over the other columns and subtracts $\log(N-1)$.
8. **`cavg(loglik, labels, beta)`** accepts at $\log \beta$ and combines the rates as in the formula. The usual mistake is dropping $1/(N-1)$.

## Where this breaks

**Too few negatives.** To trust one false accept per ten hours, test on many tens of hours that include television and music.

**An end-of-query score that trusts the words.** "Call Mom" looks complete, but the query is "call Mom at work". The short timeout cuts it off, and the cut-off rate shows it.

**Short clips.** A two-second clip pools few frames, so language scores are noisy.

## The exercise

You build `smooth_posteriors`, `detect`, `evaluate_stream`, `det_points`, `endpoint`, `endpoint_metrics`, `language_llr`, and `cavg`. Press **Run** to see the settings of Table 1, a DET plot, the endpointers on Figure 2's utterance, and a $C_\text{avg}$.

The checks work each function by hand. The planted stream must give exactly 60 false accepts per hour and an FRR of 1/6, so a detector with no hold-off or the wrong denominator fails. The last checks use the three utterances and Table 3.

## Further reading

- Warden, "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition," 2018 (https://arxiv.org/abs/1804.03209).
- Chen, Parada, and Heigold, "Small-footprint keyword spotting using deep neural networks," ICASSP, 2014 (https://research.google/pubs/small-footprint-keyword-spotting-using-deep-neural-networks/).
- Apple Machine Learning Research, "Hey Siri: An On-device DNN-powered Voice Trigger for Apple's Personal Assistant," October 2017 (https://machinelearning.apple.com/research/hey-siri).
- NIST, "The 2017 NIST Language Recognition Evaluation Plan (LRE17)," September 2017, section 3.1 (https://www.nist.gov/system/files/documents/2017/09/29/lre17_eval_plan-2017-09-29_v1.pdf).
