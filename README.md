# llm_security
This repository accompanies a bachelor’s thesis investigating the security of large language models (LLMs), with a focus on prompt injection attacks and their defense mechanisms.

##Set up
You need to download specific branch with specific protection. Main branch is only with basic word filter.

###You need to create venv:
"python3 -m venv venv"

###Download podman-compose if you do not have it
"sudo apt install podman"
"pip3 install podman-compose"

###Edit api-wrapper/environment/API_KEY and api-wrapper/environment/BASE_URL

###Run it using
'podman-compose up -d --build'

###Stop it by using
'podman-compose down'
