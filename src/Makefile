# handy target to deal with a debian bug
# https://stackoverflow.com/questions/39577984/what-is-pkg-resources-0-0-0-in-output-of-pip-freeze-command
freeze:
	pip freeze | grep -v "pkg-resources" > requirements.txt
