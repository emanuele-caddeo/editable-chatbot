from typing import Optional
from .ke_models import KnowledgeEdit


def parse_ke_command(text: str) -> Optional[KnowledgeEdit]:
    """
    Parser minimale per comandi di knowledge editing espliciti.

    Formato supportato:

    /ke edit
    subject: Canada
    prompt: The capital of {} is
    new: Toronto

    Restituisce:
    - KnowledgeEdit se il comando è valido
    - None se il messaggio NON è un comando KE
    """

    if not text:
        return None

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    if not lines:
        return None

    # Comando principale
    if lines[0].lower() != "/ke edit":
        return None

    fields = {}

    for line in lines[1:]:
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()

    required_fields = {"subject", "prompt", "new"}

    if not required_fields.issubset(fields):
        return None

    edit = KnowledgeEdit(
        subject=fields["subject"],
        prompt=fields["prompt"],
        new_object=fields["new"],
    )

    return edit
