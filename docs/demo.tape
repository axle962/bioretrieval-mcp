# Render with: vhs docs/demo.tape   (brew install vhs)
# Reads ANTHROPIC_API_KEY from .env and uses the baseline map index.
Output docs/demo.gif
Set Shell zsh
Set FontSize 15
Set Width 1300
Set Height 720
Set TypingSpeed 35ms
Set Theme "Catppuccin Mocha"

Hide
Type "source .venv/bin/activate && set -a && source .env && set +a && export BIORETRIEVAL_MAP_INDEX=artifacts/map_index_esm_baseline.npz && clear" Enter
Show
Type `python eval/demo.py "Get the KRAS (P01116) sequence, retrieve matching cryo-EM maps, and give the top hit's title and resolution."`
Sleep 500ms
Enter
Wait+Screen@120s /tool calls/
Sleep 4s
