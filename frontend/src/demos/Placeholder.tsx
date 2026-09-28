/**
 * A stand-in for a demo that is registered but not built yet, so a lesson can
 * name its demo before the demo exists. Replace the entry's import in
 * `index.ts` with the real component when it lands.
 */

import { DemoFrame } from "./kit";
import type { DemoProps } from "./index";

export default function Placeholder({ caption }: DemoProps) {
  return (
    <DemoFrame title="This demo is on its way" caption={caption}>
      <p className="px-5 py-6 text-[14px] text-ink-3">
        The interactive version of this figure isn't built yet.
      </p>
    </DemoFrame>
  );
}
