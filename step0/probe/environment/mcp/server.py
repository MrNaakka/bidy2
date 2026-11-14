"""Step-0 probe: one MCP tool that returns a PNG with a secret code drawn in it.

The code exists only in pixels (and in a token-protected endpoint the verifier uses),
so if the agent reports it correctly, the model really received the image.
"""
import io
import os
import secrets
import string

from mcp.server.fastmcp import FastMCP, Image
from mcp.server.transport_security import TransportSecuritySettings
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from starlette.requests import Request
from starlette.responses import PlainTextResponse

CODE = "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "")

mcp = FastMCP(
    "probe", host="0.0.0.0", port=8000,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@mcp.tool()
def show_card() -> Image:
    """Return a PNG image of a card with a 6-character code printed on it."""
    img = PILImage.new("RGB", (480, 200), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 64)
    d.text((40, 60), CODE, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Image(data=buf.getvalue(), format="png")


@mcp.custom_route("/_verify", methods=["GET"])
async def verify(req: Request):
    if not VERIFY_TOKEN or req.query_params.get("token") != VERIFY_TOKEN:
        return PlainTextResponse("forbidden", status_code=403)
    return PlainTextResponse(CODE)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
