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
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages garak

WORKDIR /app


