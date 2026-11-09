from app.services import spell
from app.services.pipeline import run

TEXT = "пршла весна сонце светет ласково и радосно зажурчали ручи весело защибитали птички напруду лед источнился и потрескался вразных местах около берига вада стала свабодной"

print("СПЕЛЛЕР:", spell.yandex_fix(TEXT))
print("ИТОГ:   ", run(TEXT))