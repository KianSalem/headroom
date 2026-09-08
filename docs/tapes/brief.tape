# A sentence becomes constraints. The half that says "don't" becomes ten held
# dimensions, and the loop reports a compromise instead of pretending.
Output docs/media/demo-brief.gif
Output docs/media/demo-brief.mp4
Source docs/tapes/style.tape
Set Width 1440
Set Height 780
Source docs/tapes/setup.tape

Sleep 800ms
Type `headroom master audio/synthetic/synth_02.wav mastered.wav \` Enter
Sleep 200ms
Type `  --brief "More space and width, but keep the low end tight and mono."`
Sleep 800ms
Enter
Wait+Screen@90s /wrote mastered\.wav/
Sleep 4s
