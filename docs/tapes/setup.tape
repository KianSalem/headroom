# Hidden preamble, run from the repository root: put the venv on PATH, prove
# the demo needs no credential, and use a short prompt so the frame is all
# output and no chrome.
Hide
Type "export PATH=$PWD/.venv/bin:$PATH" Enter
Type "export ANTHROPIC_API_KEY='' HEADROOM_CASSETTE_MODE=replay" Enter
Type 'export PS1="\[\e[38;5;180m\]headroom\[\e[0m\] \[\e[38;5;66m\]$\[\e[0m\] "' Enter
Type "clear" Enter
Show
