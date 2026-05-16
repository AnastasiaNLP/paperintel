from pydantic import BaseModel, Field


class HealthStatus(BaseModel):
    healthy: bool
    checks: dict[str, str] = Field(default_factory=dict)
