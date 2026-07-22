# Rejected pre-generation Phase983 chain

This directory preserves the first Phase983 protocol and engineering-
qualification artifacts.  No formal dataset generation, admission, manifest,
rows, scientific gate, holdout access, or mechanism work occurred under this
chain.

The chain was rejected on 2026-07-18 before admission because a mandatory
idempotent reread of `engineering_qualification.json` failed.  The stored
negative-test dictionaries were serialized with sorted JSON keys, while the
qualification verifier incorrectly compared their insertion order with the
in-memory test declaration order.  The three CUDA engineering smokes had
passed, and the artifact itself remained byte-for-byte unchanged, but the
existing-artifact verifier correctly blocked progress.

The verifier was changed to compare the exact key set instead of dictionary
insertion order.  That source change invalidated the old protocol's script
seal, so both artifacts were archived rather than reused.  A new protocol and
new sequential engineering qualification must be created from the corrected
source before admission.

Rejected chain identities:

- protocol self hash: `b47986c16ea220ef6452c6b5b91afb9ab1d90ee1e3b37576ac06239d6000616d`
- protocol file SHA256: `6a045eaf54af1fe9913b87f6a3245683b4658392ef8848a10f8877ecde29e51f`
- qualification self hash: `d0131f43334ac9a2ccb36949312c68b57dfcb91617563eb79775640d269941d2`
- qualification file SHA256: `40ce38b9eeac467601d8a57ddee29e9585fbccaa672a30cd6be95b4a7bb4a585`
- sealed qualification script SHA256: `82b20baf04d6ee36887ec7c06acdc2ab5931e7581f8c2588eae0f27e688ea6ec`

