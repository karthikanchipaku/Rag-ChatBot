import os
import traceback

base = os.path.dirname(__file__)
dotenv = os.path.join(base, ".env")
if os.path.exists(dotenv):
    with open(dotenv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

print("GOOGLE_API_KEY present in env:", bool(os.environ.get("GOOGLE_API_KEY")))

try:
    from google import genai
    client = genai.Client()
    print('genai module attrs:', [a for a in dir(genai) if not a.startswith('_')])
    try:
        import google.genai.models as gmodels
        print('google.genai.models attrs:', [a for a in dir(gmodels) if not a.startswith('_')])
    except Exception as e:
        print('could not import google.genai.models:', e)

    # Inspect Models class
    try:
        ModelsClass = getattr(gmodels, 'Models', None)
        if ModelsClass is not None:
            print('ModelsClass attrs:', [a for a in dir(ModelsClass) if not a.startswith('_')])
            # try calling a list-like method on ModelsClass
            for method in ('list', 'list_models', 'list_all'):
                if hasattr(ModelsClass, method):
                    try:
                        print(f"Calling Models.{method}()")
                        res = getattr(ModelsClass, method)()
                        print('Result type:', type(res))
                        break
                    except Exception as ex:
                        print(f"Models.{method}() failed:")
                        import traceback
                        traceback.print_exc()
    except Exception as e:
        print('Error inspecting ModelsClass:', e)

    # Try a few candidate list functions
    tried = False
    for cand in ('list_models', 'models.list_models', 'models.list', 'get_models'):
        try:
            if cand == 'list_models' and hasattr(client, 'list_models'):
                models = list(client.list_models())
            elif cand == 'models.list_models' and hasattr(genai, 'models') and hasattr(genai.models, 'list_models'):
                models = list(genai.models.list_models())
            elif cand == 'models.list' and hasattr(genai, 'models') and hasattr(genai.models, 'list'):
                models = list(genai.models.list())
            elif cand == 'get_models' and hasattr(genai, 'get_models'):
                models = list(genai.get_models())
            else:
                continue
            print(f"Found models via {cand}:")
            for m in models[:50]:
                print(m.name, getattr(m, 'supported_methods', None))
            tried = True
            break
        except Exception:
            import traceback
            print(f"Attempt {cand} failed:")
            traceback.print_exc()
    if not tried:
        print('No candidate list_models method succeeded.')
    # Inspect client and try client.models.list()
    try:
        print('client attrs:', [a for a in dir(client) if not a.startswith('_')])
        if hasattr(client, 'models'):
            try:
                models = list(client.models.list())
                print('client.models.list() succeeded:')
                for m in models[:50]:
                    print(m.name, getattr(m, 'supported_methods', None))
            except Exception:
                print('client.models.list() failed:')
                import traceback
                traceback.print_exc()
    except Exception as e:
        print('Error inspecting client:', e)
except Exception:
    print("Exception while listing models:")
    traceback.print_exc()
