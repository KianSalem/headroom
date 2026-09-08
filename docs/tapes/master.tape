# The product surface: point it at a delivery target and it closes the loop.
# No credential, no network -- the default system is arithmetic.
Output docs/media/demo-master.gif
Output docs/media/demo-master.mp4
Source docs/tapes/style.tape
Set Width 1440
Set Height 420
Source docs/tapes/setup.tape

Sleep 800ms
Type `headroom master audio/synthetic/synth_01.wav mastered.wav --target club`
Sleep 800ms
Enter
Wait+Screen@90s /wrote mastered\.wav/
Sleep 3s
