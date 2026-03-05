eval $(awk -F' = ' '{gsub(/"/, "", $2); print $1"="$2}' \
  "$(python -c "import methylseqnet; print(methylseqnet.__path__[0])")/../configs/paths.toml")