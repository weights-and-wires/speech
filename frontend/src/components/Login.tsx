import Mark from "./Mark";

export default function Login() {
  const returnTo = new URL(window.location.href);
  returnTo.pathname = window.location.pathname;
  returnTo.search = window.location.search;
  returnTo.hash = window.location.hash;
  const portal = new URL("https://auth.ifkash.dev/");
  portal.searchParams.set("rd", returnTo.href);

  return (
    <div className="flex min-h-full items-center justify-center bg-canvas px-5 py-12">
      <div className="w-full max-w-[22rem]">
        <Mark size={32} />
        <h1 className="mt-5 font-serif text-[2rem] font-semibold leading-tight tracking-[-0.015em] text-ink">
          Speech and audio
        </h1>
        <p className="mt-2 font-serif text-[1.05rem] leading-relaxed text-ink-3">
          Lessons you can hear, from a sampled waveform to models that
          listen and speak, and every one ends with code you write yourself.
        </p>
        <a href={portal.href} className="btn btn-primary mt-8 flex min-h-10 w-full items-center justify-center text-[14px]!">
          Sign in with Authelia
        </a>
      </div>
    </div>
  );
}
