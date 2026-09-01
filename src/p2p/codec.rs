use crate::p2p::types::P2pMessage;
use anyhow::{bail, Context, Result};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

const MAX_FRAME_SIZE: usize = 16 * 1024 * 1024; // 16 MB maximum frame size
const MIN_FRAME_SIZE: usize = 2; // Minimal JSON payload "{}"

/// Asynchronously write a length-delimited P2pMessage to a stream.
pub async fn write_message<W: AsyncWrite + Unpin>(writer: &mut W, msg: &P2pMessage) -> Result<()> {
    let payload = serde_json::to_vec(msg).context("Failed to serialize P2pMessage")?;
    let len = payload.len();
    if len > MAX_FRAME_SIZE {
        bail!("P2pMessage exceeds maximum frame size: {} > {}", len, MAX_FRAME_SIZE);
    }

    let len_bytes = (len as u32).to_be_bytes();
    writer.write_all(&len_bytes).await?;
    writer.write_all(&payload).await?;
    writer.flush().await?;
    Ok(())
}

/// Asynchronously read a length-delimited P2pMessage from a stream.
pub async fn read_message<R: AsyncRead + Unpin>(reader: &mut R) -> Result<P2pMessage> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;

    if !(MIN_FRAME_SIZE..=MAX_FRAME_SIZE).contains(&len) {
        bail!("Incoming frame size outside valid bounds ({}..={}): got {}", MIN_FRAME_SIZE, MAX_FRAME_SIZE, len);
    }

    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload).await?;

    let msg: P2pMessage = serde_json::from_slice(&payload).context("Failed to deserialize P2pMessage")?;
    Ok(msg)
}
