"""Offline API contract fixture. NEVER use these results as an efficacy claim."""
from ..backend import ModelBackend
from ..toy import ToyBackend
from ..types import ModelResult
from .common import canonical,strict_json


class SuiteToyBackend(ModelBackend):
    def cache_identity(self): return {"provider":"suite-toy","fixture_version":1}

    def generate(self,*,model,messages):
        developer=messages[0]["content"]
        last=messages[-1]["content"]
        if "REFINER_COMPILE" in developer:
            result=ToyBackend().generate(model="toy",messages=[{"role":"developer","content":"compiler"},
                {"role":"user","content":"observed baseline errors"}])
        elif "REFINER_PREFERENCE" in developer:
            result=ModelResult("- Prefer concise, direct responses consistent with the preferred demonstrations.")
        elif '"accept":true' in developer:
            result=ModelResult('{"accept":true}')
        else:
            adapted=developer
            for msg in messages:
                if msg["content"].startswith("REFINER_MATERIAL_JSON\n"):
                    material=strict_json(msg["content"].split("\n",1)[1])
                    if material["rules"]: adapted+="\nREFINEMENT RULES:\n"+material["rules"]
                    demos=[{"input":x["input"],"correct_output":x["target"]} for x in material["examples"]]
                    adapted+="\n<refiner_examples_json>\n"+canonical(demos)+"\n</refiner_examples_json>"
            result=ToyBackend().generate(model="toy",messages=[{"role":"developer","content":adapted},{"role":"user","content":last}])
        return ModelResult(result.text,sum(len(x["content"].split()) for x in messages),max(1,len(result.text.split())),
                           {"provider":"suite-toy","synthetic":True})
