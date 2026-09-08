# Measurement first, then enforcement: the chain order is imposed by the
# system, not trusted from the input.
Output docs/media/demo-invariants.gif
Output docs/media/demo-invariants.mp4
Source docs/tapes/style.tape
Set Width 1440
Set Height 420
Source docs/tapes/setup.tape

Sleep 800ms
Type `headroom compare audio/synthetic/synth_01.wav --preset club`
Sleep 600ms
Enter
Wait+Screen@60s /TOO LOW/
Sleep 2s
Type `headroom render audio/synthetic/synth_02.wav docs/media/out-of-order-chain.json ordered.wav`
Sleep 600ms
Enter
Wait+Screen@60s /wrote ordered\.wav/
Sleep 4s
