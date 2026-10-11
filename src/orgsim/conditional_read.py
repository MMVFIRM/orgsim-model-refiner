"""Evidence-limited, conditional API read predictor.

No Dolibarr values are persisted here. This is a general-purpose, role-scoped
schema/endpoint predictor built from prior *development* observations. It
explicitly distinguishes array emptiness, projection, permission, and type
polymorphism. A new endpoint without evidence may be left unpredicted.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any
from urllib.parse import urlsplit, parse_qs

TYPE_KINDS = {'array','object','string','number','boolean','null'}


def _profile(shape:dict) -> dict:
    if not isinstance(shape,dict):return {}
    if 'kind' in shape:
        keys = shape.get('first_keys', shape.get('keys', []))
        types = shape.get('first_types', shape.get('types', {}))
        return {'kind':shape['kind'], 'length':shape.get('length'),
                'item_kind':shape.get('first_item_kind'),
                'field_keys':keys,'field_types':types}
    return {}


class ConditionalReadModel:
    """Fits only on observations explicitly supplied BEFORE candidate freezing.

    Operations are identified by route *template* (not data-bearing actual URLs).
    The model NEVER consumes evaluation results during an evaluation run.
    """
    def __init__(self, discovery:dict, development:dict|None=None):
        self.discovery=discovery
        self.development=development or {}
        self.family_profiles={}
        self.route_profiles=defaultdict(list)
        self.type_evidence=defaultdict(lambda:defaultdict(Counter))
        self.permission_outcomes=defaultdict(Counter)
        self.suffix_profiles=defaultdict(list)
        self.parameter_suppressions=defaultdict(set)
        for observation in discovery.get('probes', []):
            family=observation.get('family')
            route=observation.get('path_template')
            status=observation.get('status')
            shape=_profile(observation.get('shape',{}))
            if not family or not route or not shape or not isinstance(status,int):continue
            self._add(family,route,status,shape)
        for observation in self.development.get('cases', []):
            family=observation.get('family')
            route=observation.get('route_template')
            truth=observation.get('observation',{})
            if not family or not route or not isinstance(truth.get('status'),int):continue
            # Prior holdout outcomes are DEVELOPMENT evidence for a NEW plan only.
            self._add(family,route,truth['status'],truth)
        self._infer_parameter_effects()

    def _add(self,family:str,route:str,status:int,shape:dict):
        parsed=urlsplit(route)
        signature=parsed.path
        payload={'status':status,**shape}
        self.route_profiles[signature].append(payload)
        self.permission_outcomes[signature][status]+=1
        last=signature.rsplit('/',1)[-1]
        if last and not last.startswith('{') and signature.count('/')>=3:
            self.suffix_profiles[last].append((family,status,payload.get('kind')))
        for key,ty in payload.get('field_types',{}).items():
            if ty in TYPE_KINDS and ty!='null':
                self.type_evidence[family][key][ty]+=1
        if parsed.path=='/'+family:
            profile=self.family_profiles.get(family)
            # Prefer the earlier full-list anchor to projected partial schemas.
            if profile is None or (len(payload.get('field_keys',[]))>len(profile.get('field_keys',[]))):
                self.family_profiles[family]=payload

    def _infer_parameter_effects(self):
        """Induce explicit field omissions from DIFFERENTIAL development evidence.

        Only a nonempty projected response can demonstrate that a requested
        field was omitted. We also require the field to have been observed on
        the same family in the unmodified collection; an unobserved field is
        not evidence of a query-dependent suppression.
        """
        for row in self.development.get('cases',[]):
            family=row.get('family');route=row.get('route_template','')
            obs=row.get('observation',{})
            if family not in self.family_profiles or obs.get('status')!=200:
                continue
            options=parse_qs(urlsplit(route).query)
            props=options.get('properties')
            if not props or obs.get('kind')!='array' or obs.get('length',0)<=0:
                continue
            if 'item_keys' not in obs:continue
            selected=set(props[0].split(','))
            actual=set(obs['item_keys'])
            observed_base=set(self.family_profiles[family].get('field_keys',[]))
            missing=(selected-actual)&observed_base
            for key,values in options.items():
                if key in ('properties','limit','page','sortfield','sortorder'):continue
                for value in values:
                    for field in missing:
                        self.parameter_suppressions[(family,key,value)].add(field)

    def suppressed_fields(self,family:str,params:dict)->set[str]:
        missing=set()
        for param,values in params.items():
            for value in values:
                missing.update(self.parameter_suppressions.get((family,param,value),()))
        return missing

    def evidence_summary(self)->dict:
        return {
            'source':'discovery_and_revealed_development_only',
            'observed_route_patterns':len(self.route_profiles),
            'observed_status_outcomes':sum(sum(c.values()) for c in self.permission_outcomes.values()),
            'polymorphic_type_fields':{family:{key:sorted(ct) for key,ct in values.items() if len(ct)>1}
                for family,values in self.type_evidence.items() if any(len(c)>1 for c in values.values())},
            'learned_query_suppressions':[
                {'family':f,'parameter':k,'value':v,'omitted_fields':sorted(fields)}
                for (f,k,v),fields in sorted(self.parameter_suppressions.items())],
            'role_scope':'same_readonly_API_user',
            'conditional_predictions_are_provisional':True}

    def observed_type(self, family:str, key:str) -> str|list[str]|None:
        counts=self.type_evidence[family].get(key,Counter())
        if not counts:return None
        kinds=sorted(counts)
        return kinds[0] if len(kinds)==1 else kinds

    def predict(self, q:Any, contract:dict) -> dict:
        route=q.route;parts=urlsplit(route);params=parse_qs(parts.query)
        fam=q.family;base=self.family_profiles.get(fam,{})
        prev=self.route_profiles.get(parts.path,[])
        pred={}
        # Role and endpoint specific. Do NOT generalize a 403 from one path
        # to another, even if the two endpoints have similar names.
        statuses={x['status'] for x in prev}
        if len(statuses)==1 and 403 in statuses:
            return {'status':403,'kind':'object'}
        if len(statuses)==1 and 404 in statuses:
            # 404 may be identity-dependent. Do not export as universal rule.
            return {}
        if len(statuses)==1 and 200 in statuses:
            pred['status']=200
            kinds={x.get('kind') for x in prev if x.get('kind')}
            if len(kinds)==1:pred['kind']=next(iter(kinds))
        # Cross-resource analogy: require at least TWO distinct family witnesses
        # for generic relationship suffixes, OR a same-family witness for a
        # key-index variant of exactly the same relation.
        if not prev and not pred:
            tail=parts.path.rsplit('/',1)[-1]
            witnesses=self.suffix_profiles.get(tail,[])
            applicable=(len({x[0] for x in witnesses})>=2 or
                        ('/ref/{ref}/' in route and any(x[0]==fam for x in witnesses)))
            if applicable and len({(x[1],x[2]) for x in witnesses})==1:
                status,kind=witnesses[0][1:]
                if status==200 and kind in {'array','object'}:
                    pred={'status':status,'kind':kind}
        # Collections use endpoint structure but not fixed record membership.
        is_collection=parts.path=='/'+fam
        if is_collection and base.get('status')==200:
            pred.update({'status':200,'kind':'array'})
            if 'ids_only' in params and params['ids_only']==['1']:
                pred['item_kind']='string'
                if 'limit' in params and set(params)<= {'limit','page','ids_only'} and params.get('page',['0'])==['0'] and base.get('length') is not None:
                    pred['length']=min(int(params['limit'][0]),base['length'])
                return pred
            status_filter=any(k in params for k in ('status','mode','category','sqlfilters','thirdparty_ids',
                                                    'variant_filter','supplier'))
            # If the selector is potentially empty, don't hallucinate a schema
            # or its size. Presence of fields is conditional on nonempty result.
            if status_filter:return pred
            # Pagination information is only extrapolated when the prior
            # collection count was exact and the only changed params are known.
            if set(params)<= {'limit','page','properties','sortfield','sortorder','withLines'}:
                if 'limit' in params and isinstance(base.get('length'),int):
                    try:
                        n=int(params['limit'][0]);page=int(params.get('page',['0'])[0])
                        if 0<=n<=20 and 0<=page<=100:
                            pred['length']=max(0,min(n,base['length']-page*n))
                    except (ValueError,TypeError):pass
            if q.special=='reverse' and (base.get('length') or 0)>=2:
                pred['reverse_matches']=True
            # Even a predictable empty page must not carry a phantom item schema.
            if pred.get('length')==0:return pred
            props=params.get('properties',[])
            if props:
                selected=set(q.selected)&set(props[0].split(','))
                selected-=self.suppressed_fields(fam,params)
                known={k for k in selected if self.observed_type(fam,k)}
                if known and len(known)==len(selected):
                    pred['item_keys']=sorted(known)
                types={k:self.observed_type(fam,k) for k in sorted(known)}
                if types:pred['field_types']=types
            return pred
        # Alternate-key fetches whose existence semantics have been observed.
        if '{ref}' in route:
            if any(x['status']==200 and x.get('kind')=='object' for x in prev):
                pred={'status':200,'kind':'object'}
                if q.special=='ref_match':pred['ref_matches']=True
            return pred
        # Related records: only reuse same endpoint type for the same role;
        # a type seen once is not treated as universal for every field.
        if prev and pred.get('kind')=='object' and q.selected:
            omitted=self.suppressed_fields(fam,params)
            selected=[k for k in q.selected if k not in omitted]
            typ={k:self.observed_type(fam,k) for k in selected}
            pred['field_types']={k:v for k,v in typ.items() if v}
        elif prev and pred.get('kind')=='array' and q.selected:
            if any(x.get('length',0) for x in prev):
                typ={k:self.observed_type(fam,k) for k in q.selected}
                pred['field_types']={k:v for k,v in typ.items() if v}
        return pred
