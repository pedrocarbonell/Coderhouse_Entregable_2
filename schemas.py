from enum import Enum
from pydantic import BaseModel, Field


class NivelCriticidad(str, Enum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"

class EntityExtraction(BaseModel):
    tecnologias: list[str] = Field(min_length=1, description="Tecnologias")
    nivel_de_criticidad: NivelCriticidad = Field(description="Nivel de criticidad")
    resumen_tecnico: str = Field(description="Resumen tecnico")