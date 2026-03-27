FROM docker.io/kalilinux/kali-rolling
RUN apt update && apt install -y \
    python3 \
    python3-pip \
    nmap \
    curl \
    tcpdump \
    netcat-traditional \
    iproute2 \
    iputils-ping \
    vim \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN pip install requests --break-system-packages

WORKDIR /app

RUN pip install requests --break-system-packages

