from dataclasses import dataclass


@dataclass
class KnowledgeEdit:
    """
    Rappresentazione strutturata di una modifica di conoscenza.

    NON è una tripla RDF.
    È una istruzione operativa per ROME.
    """
    subject: str
    prompt: str          # Deve contenere '{}'
    new_object: str

    def render(self) -> str:
        """
        Rappresentazione leggibile per conferma utente.
        Esempio:
        'The capital of Canada is Toronto'
        """
        return f"{self.prompt.format(self.subject)} {self.new_object}"

    def is_valid(self) -> None:
        """
        Valida semanticamente l'edit.
        Lancia ValueError se non valido.
        """
        if "{}" not in self.prompt:
            raise ValueError("Prompt must contain '{}' placeholder")

        if not self.subject or len(self.subject.strip()) < 2:
            raise ValueError("Subject is too short or empty")

        if not self.new_object or len(self.new_object.strip()) < 1:
            raise ValueError("New object is empty")
