FROM rust:alpine AS builder
RUN apk add --no-cache musl-dev
WORKDIR /app
COPY Cargo.toml Cargo.lock build.rs ./
COPY src       ./src
COPY resources ./resources
# target-cpu=haswell enables all instruction extensions available on the
# competition hardware (Mac Mini Intel 2014, Haswell): AVX2, FMA3, SSE4.1/4.2,
# BMI1/2, POPCNT.  More complete than listing individual features manually.
# The musl target produces a fully static binary with no libc dependency.
ENV RUSTFLAGS="-C target-cpu=haswell"
RUN cargo build --release --features ivf --target x86_64-unknown-linux-musl

FROM busybox:musl
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/rinha /rinha
EXPOSE 8080
ENTRYPOINT ["/rinha"]
