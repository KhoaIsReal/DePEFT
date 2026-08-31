# Multi-stage Docker build for DePEFT Node & Miner
FROM rust:1.80-bullseye as builder

WORKDIR /usr/src/depeft
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

# Minimal Runtime Image
FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y ca-certificates libssl-dev curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/depeft/target/release/depeft /usr/local/bin/depeft

ENV RUST_LOG=info
EXPOSE 8545 9000

ENTRYPOINT ["depeft"]
CMD ["node", "start", "--port", "8545", "--p2p-port", "9000"]
