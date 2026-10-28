# Jev has no job in Record Scan today

"Typesafe's Jev" is Jev, a model from TypeSafe AI, a San Francisco lab that launched it on 15 September 2026 ([TypeSafe blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)). It answers typed questions with probabilities instead of generating text. Record Scan should not adopt it. The project's two model calls are "name the record in this photo" and "embed this frame in under 30 ms on a phone". Jev cannot do the first, because it generates no text. It cannot do the second, because it is a hosted API with 70 to 500 ms round trips. Its image support, the only part that touches covers, is flagged by the vendor as not ready for production. The one place it could help later is re-ranking the top few retrieval candidates on hard misses. That is worth a one-afternoon test only if Task 7's miss gallery shows near-duplicate confusion as a large share of misses, and only once images leave the experimental flag.

## What Jev is

Jev takes a `state` (text, JSON, or images) plus named questions, and returns one typed answer per question ([API reference](https://docs.typesafe.ai/api)). There are four question types ([API reference](https://docs.typesafe.ai/api)):

- `noul`: a yes/no statement, answered with a probability from 0 to 1.
- `choice`: picks one of up to 255 named options and returns a probability for each ([Choice docs](https://docs.typesafe.ai/primitives/choice.md)).
- `score`: rates content on an ordered rubric of up to 10 levels.
- `box`: returns one bounding box in an image.

It does not write free text: "No text generation, no parsing" ([introduction](https://docs.typesafe.ai/introduction.md)). TypeSafe pitches it for classification, routing, scoring, extraction and guardrails on LLM output, and leaves chat and open-ended generation to LLMs ([launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev)). It is an API only. The models page lists no weights, and the legal page covers terms, a data processing agreement and a promise not to train on user data, with nothing about self-hosting ([models](https://docs.typesafe.ai/models.md); [legal](https://docs.typesafe.ai/legal.md)).

## Maturity

Jev is six weeks old.

| Item | Fact | Source |
|---|---|---|
| Launch | 15 September 2026, early access with a waitlist | [launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev) |
| Model versions | `jev-1.13.0`, then `jev-1.14.0` on 14 October 2026; `jev-latest` points at 1.14 | [models](https://docs.typesafe.ai/models.md) |
| Deprecation notice | 60 days for 1.13, 30 days for 1.14 | [Jev 1.14 post](https://typesafe.ai/blog/jev-1-14) |
| API version | OpenAPI 0.2.0 | [API reference](https://docs.typesafe.ai/api) |
| JS SDK | `@typesafe-ai/sdk` 0.7.0, 14 October 2026, MIT | [npm registry](https://registry.npmjs.org/@typesafe-ai/sdk) |
| Python SDK | `typesafe-sdk` 0.7.5, 14 October 2026, MIT, nine releases since 9 September | [PyPI](https://pypi.org/project/typesafe-sdk/) |
| SDK repos | created 4 September 2026, last push 14 October 2026, 22 and 6 open issues | [typesafe-sdk-js](https://github.com/typesafe-ai/typesafe-sdk-js); [typesafe-sdk-python](https://github.com/typesafe-ai/typesafe-sdk-python) |
| Price | $0.042 per million input tokens, output free; the vendor says it cannot prove the price isn't subsidized | [launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev) |
| Latency | 70 to 500 ms end to end, measured from the founders' laptops on the US West Coast | [launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev) |

Every speed, cost and accuracy figure comes from TypeSafe. The launch post's benchmark FAQ entry has no answer, and the homepage's "193.6x faster, 444.6x cheaper" headline does not match its own per-call figures of 0.114 s and $0.000081 against 8.566 s and $0.013880, which work out to about 75x and 171x ([typesafe.ai](https://typesafe.ai)).

Images are the weak spot for this project. Image input and `box` sit behind a `TypeSafe-Danger-Zone-Images: true` header. The docs say they "aren't yet up to our standards to call ready for production" and that formats, answers and accuracy may change between releases ([images docs](https://docs.typesafe.ai/images.md)). The 1.14 post says there is "still way too much jaggedness for us to vouch for these for production use cases", and that every model, Jev included, does worse on images than on text ([Jev 1.14 post](https://typesafe.ai/blog/jev-1-14)). Images must be PNG or JPEG, 280 to 1568 px on each side, and the API rejects out-of-range images instead of resizing them ([images docs](https://docs.typesafe.ai/images.md)). The 1.13 limitations page also says Jev "leans toward the option that comes first" in a choice ([jev-1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)), which matters for any candidate re-ranker.

## Where it would sit in this repo

The only live model call is `web/src/app/api/openai/image_detection/route.ts`. It sends the photo to `gpt-5-mini` and asks for an `album_name` string and an `artist_name` list of strings, which `web/src/app/page.tsx` then searches on Deezer. Jev cannot return an album name it was not given, so it cannot replace this route. The plan already keeps that route only as the fallback behind the retrieval service (Task 10 in `docs/superpowers/plans/2026-10-24-identifier-plan.md`).

The plan's core does not call a hosted model at all. It embeds frames with a local backbone and searches an HNSW index (`pipeline/retrieval/embed.py`, `build_index.py`, `identify.py`), heading for an on-device student held to a 30 ms p95 bar (D13). D14's crate-flipping case needs the app to find the front record in real time and stay quiet between records. A 70 to 500 ms network call per frame misses both.

Three Jev features look relevant on paper. None earns a place now:

- `box` to find the cover in a crate frame. Task 7's crop study already tests OpenCV contour rectify and Apple's `VNDetectRectanglesRequest`, both local and free. Jev's `box` is the experimental one.
- `noul` for "is there a record in this frame". The gold set's 100 to 200 negative frames (H2) are there to calibrate a retrieval score threshold for exactly this, with no extra service.
- `choice` over the top five retrieval candidates, with their cover thumbnails as images in the criteria. This is the one idea with some merit: a second opinion when two pressings look alike. It still needs the experimental image flag. Most gallery covers are 500 px originals (D10), above Jev's 280 px floor, but the smaller originals would need upscaling.

The pipeline's metadata tools (`pipeline/metadata/`) match and diff MusicBrainz dumps with deterministic code and SQLite FTS5. Nothing there needs a probabilistic classifier.

## Recommendation

Do not add Jev. It solves a problem Record Scan does not have: turning free-form LLM output into structured decisions at scale. Record Scan's decisions come from vector distances it already controls, and its hard constraints are on-device latency and offline use.

Revisit once, after Task 7, if both hold. First, the miss gallery shows that a large share of remaining misses have the right cover in the top five, lost to near-duplicate confusion. Second, TypeSafe has moved images out of the danger-zone flag. Then score `choice` over the top five thumbnails on the gold set's misses. At $0.042 per million input tokens the test costs cents. Check its first-option bias by shuffling candidate order. Keep it only if it lifts r1 on the held-up case, and only as a server-side fallback, never in the real-time crate path.
