# NOTE: THIS DOCKERFILE IS GENERATED VIA "apply-templates.sh"
FROM debian:bookworm-slim AS python-build

ENV PATH /usr/local/bin:$PATH
ENV LANG C.UTF-8
ENV GPG_KEY A035C8C19219BA821ECEA86B64E628F8D684696D
ENV PYTHON_VERSION 3.11.11
ENV PYTHON_SHA256 2a9920c7a0cd236de33644ed980a13cbbc21058bfdc528febb6081575ed73be3

# Combine RUN commands and clean up in the same layer
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        netbase \
        tzdata \
        dpkg-dev \
        gcc \
        gnupg \
        libbluetooth-dev \
        libbz2-dev \
        libc6-dev \
        libdb-dev \
        libffi-dev \
        libgdbm-dev \
        liblzma-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        make \
        tk-dev \
        uuid-dev \
        wget \
        xz-utils \
        zlib1g-dev \
    && wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz" \
    && echo "$PYTHON_SHA256 *python.tar.xz" | sha256sum -c - \
    && wget -O python.tar.xz.asc "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc" \
    && GNUPGHOME="$(mktemp -d)" \
    && export GNUPGHOME \
    && gpg --batch --keyserver hkps://keys.openpgp.org --recv-keys "$GPG_KEY" \
    && gpg --batch --verify python.tar.xz.asc python.tar.xz \
    && gpgconf --kill all \
    && rm -rf "$GNUPGHOME" python.tar.xz.asc \
    && mkdir -p /usr/src/python \
    && tar --extract --directory /usr/src/python --strip-components=1 --file python.tar.xz \
    && rm python.tar.xz

# Build Python
WORKDIR /usr/src/python
RUN set -eux; \
    gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)" \
    && ./configure \
        --build="$gnuArch" \
        --enable-loadable-sqlite-extensions \
        --enable-optimizations \
        --enable-option-checking=fatal \
        --enable-shared \
        --with-lto \
        --with-ensurepip \
    && nproc="$(nproc)" \
    && EXTRA_CFLAGS="$(dpkg-buildflags --get CFLAGS)" \
    && LDFLAGS="$(dpkg-buildflags --get LDFLAGS)" \
    && LDFLAGS="${LDFLAGS:--Wl},--strip-all" \
    && make -j "$nproc" \
        "EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" \
        "LDFLAGS=${LDFLAGS:-}" \
    && rm python \
    && make -j "$nproc" \
        "EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" \
        "LDFLAGS=${LDFLAGS:--Wl},-rpath='\$\$ORIGIN/../lib'" \
        python \
    && make install \
    && pip3 install --no-cache-dir 'setuptools==65.5.1' wheel

# Create a new stage for librealsense build
FROM python-build AS librealsense-builder

ARG LIBRS_VERSION
ENV DEBIAN_FRONTEND=noninteractive

# Install only necessary build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libssl-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Download and build librealsense
WORKDIR /usr/src
RUN curl -L https://github.com/IntelRealSense/librealsense/archive/refs/tags/v${LIBRS_VERSION}.tar.gz | tar xz \
    && cd librealsense-${LIBRS_VERSION} \
    && mkdir build && cd build \
    && cmake \
        -DCMAKE_C_FLAGS_RELEASE="${CMAKE_C_FLAGS_RELEASE} -s" \
        -DCMAKE_CXX_FLAGS_RELEASE="${CMAKE_CXX_FLAGS_RELEASE} -s" \
        -DCMAKE_INSTALL_PREFIX=/opt/librealsense \
        -DPYTHON_EXECUTABLE=$(which python3.11) \
        -DBUILD_PYTHON_BINDINGS:bool=true \
        -DCMAKE_BUILD_TYPE=Release \
        -DFORCE_RSUSB_BACKEND=true \
        -DBUILD_EXAMPLES=false \
        -DBUILD_GRAPHICAL_EXAMPLES=false \
        ../ \
    && make -j$(($(nproc)-1)) all \
    && make install
RUN mkdir -p /usr/local/lib/python3.11/site-packages/pyrealsense2/
RUN cp -r /usr/src/librealsense-${LIBRS_VERSION}/wrappers/python/pyrealsense2/__init__.py /usr/local/lib/python3.11/site-packages/pyrealsense2/ && \
	cp -r /usr/src/librealsense-${LIBRS_VERSION}/build/Release/pyrealsense2* /usr/local/lib/python3.11/site-packages/pyrealsense2/
# Final stage
FROM debian:bookworm-slim

# Copy only necessary files from previous stages
COPY --from=python-build /usr/local /usr/local
COPY --from=librealsense-builder /opt/librealsense /usr/local/librealsense
COPY --from=librealsense-builder /usr/src/librealsense-*/config/99-realsense-*.rules /etc/udev/rules.d/
COPY --from=librealsense-builder /usr/local/lib/python3.11/site-packages/pyrealsense2 /usr/local/lib/python3.11/site-packages/pyrealsense2

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libusb-1.0-0 \
    udev \
    libgtk-3-0 \
    libglfw3 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=$PYTHONPATH:/usr/local/lib
ENV LD_LIBRARY_PATH=/usr/local/librealsense/lib
ENV PATH=/usr/local/librealsense/bin:$PATH

WORKDIR /app
RUN mkdir -p /app/out
RUN python3.11 -c "import pyrealsense2"
CMD ["rs-enumerate-devices", "--compact"]