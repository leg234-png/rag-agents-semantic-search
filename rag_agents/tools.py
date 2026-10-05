"""Outils mis à disposition de l'agent. Chaque outil a un schéma d'arguments Pydantic :
des arguments invalides renvoient une erreur lisible à l'agent, qui peut se corriger."""
from __future__ import annotations

from typing import Callable, Literal

from pydantic import BaseModel, Field

from .retrieval import Retriever

# Taux indicatifs repris de la base documentaire (paiement.md)
RATES = {"USD": 1.08, "GBP": 0.85, "CHF": 0.95}


class SearchArgs(BaseModel):
    query: str = Field(description="Requête de recherche reformulée, précise")
    k: int = Field(4, ge=1, le=8)


class CancellationArgs(BaseModel):
    price_eur: float = Field(gt=0, description="Prix total payé en euros")
    days_before_departure: int = Field(ge=0, description="Nombre de jours entre l'annulation et le départ")
    fare: Literal["basic", "classic", "flex"] = "classic"


class ConvertArgs(BaseModel):
    amount_eur: float
    currency: Literal["USD", "GBP", "CHF"]


def cancellation_fee(args: CancellationArgs) -> dict:
    """Barème d'annulation (annulation.md). Les tarifs Flex sont remboursables jusqu'à J-1."""
    if args.fare == "flex" and args.days_before_departure >= 1:
        rate = 0.0
    elif args.days_before_departure > 30:
        rate = 0.10
    elif args.days_before_departure >= 15:
        rate = 0.30
    elif args.days_before_departure >= 7:
        rate = 0.50
    else:
        rate = 1.0
    fee = round(args.price_eur * rate, 2)
    return {"fee_rate": rate, "fee_eur": fee, "refund_eur": round(args.price_eur - fee, 2),
            "source": "annulation#bareme-des-frais-d-annulation"}


def convert(args: ConvertArgs) -> dict:
    return {"amount": round(args.amount_eur * RATES[args.currency], 2), "currency": args.currency,
            "rate": RATES[args.currency], "source": "paiement#devises-et-frais-bancaires"}


class Toolbox:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.specs: dict[str, tuple[type[BaseModel], Callable, str]] = {
            "search_docs": (SearchArgs, self._search,
                            "Recherche dans la base documentaire (bagages, annulation, retards, assurance...)."),
            "compute_cancellation_fee": (CancellationArgs, cancellation_fee,
                                         "Calcule les frais d'annulation et le remboursement selon le barème."),
            "convert_currency": (ConvertArgs, convert, "Convertit un montant en euros en USD, GBP ou CHF."),
        }

    def _search(self, args: SearchArgs) -> dict:
        hits = self.retriever.search(args.query, args.k)
        return {"results": [{"id": c.id, "section": f"{c.title} - {c.section}", "text": c.text} for c, _ in hits]}

    def describe(self) -> str:
        return "\n".join(f"- {name}({', '.join(schema.model_fields)}) : {desc}"
                         for name, (schema, _, desc) in self.specs.items())

    def call(self, name: str, raw_args: dict) -> dict:
        if name not in self.specs:
            return {"error": f"outil inconnu : {name}"}
        schema, fn, _ = self.specs[name]
        try:
            return fn(schema.model_validate(raw_args))
        except Exception as err:  # renvoyé à l'agent comme observation
            return {"error": f"arguments invalides pour {name} : {err}"}
