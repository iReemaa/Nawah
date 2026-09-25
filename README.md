# Nawah — ربط الواجهة بالوكيل

هذا المجلد يحوّل خط أنابيب الوكيل في `nawah_professional_v3.ipynb` إلى خادم
بايثون قائم بذاته (FastAPI)، وواجهة `nawah_connected.html` معدّلة لتتصل به
فعليًا بدل البيانات المحلية المزيّفة.

## الملفات

| الملف | الوصف |
|---|---|
| `nawah_pipeline.py` | نفس منطق الوكيل بالضبط (Researcher → Verifier1 → Planner → Verifier2 → Orchestrator + طبقة المحادثة)، منقول من النوتبوك بدون أي تعديل على السلوك، فقط أُزيلت اعتماديات Colab |
| `server.py` | خادم FastAPI يكشف الوكيل عبر REST: `/api/plan` و`/api/chat` و`/api/health` |
| `nawah_connected.html` | نسخة من واجهتك الأصلية (`Pasted.html`) بعد تعديل `buildPlan()`/`go()` لتستدعي الخادم الحقيقي، مع سقوط تلقائي (fallback) للعرض التجريبي المحلي إذا تعذّر الاتصال |
| `requirements.txt` | حزم بايثون المطلوبة |
| `.env.example` | نموذج متغيرات البيئة |
| `Dockerfile` | لتغليف الخادم ونشره على أي منصة حاويات |

## 1) التشغيل محليًا

```bash
cd nawah_server
python -m venv .venv && source .venv/bin/activate   # اختياري لكن يُنصح به
pip install -r requirements.txt
cp .env.example .env
# افتحي .env وضعي OPENAI_API_KEY الحقيقي

mkdir -p data
# انسخي فيه: knowledge_2.json + quality_issues.json + manifest.json
# (نفس الملفات التي كان النوتبوك يطلب رفعها إلى Colab)

uvicorn server:app --reload --port 8000
```

جرّبي:
```bash
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/plan \
  -H "Content-Type: application/json" \
  -d '{"business_idea":"مقهى قهوة مختصة في الرياض","nationality":"saudi"}'
```

## 2) فتح الواجهة محليًا

افتحي `nawah_connected.html` مباشرة في المتصفح (double-click أو أي خادم
ملفات ثابت). القيمة `API_BASE = "http://localhost:8000"` في أعلى قسم
JavaScript بالملف تشير للخادم المحلي تلقائيًا — لا حاجة لأي تعديل إذا كان
الخادم يعمل على نفس الجهاز والمنفذ 8000.

اكتبي فكرة عمل في مربع البحث بالأعلى واضغطي زر الإرسال: سترين مذكرة
"المستشار الحقيقي" تظهر فوق قائمة الخطوات، مبنية فعليًا من استجابة
`/api/plan`. إذا تعذّر الوصول للخادم (غير مشغَّل، رابط خاطئ، CORS...)
تظهر رسالة خطأ واضحة، وتبقى الخطة التجريبية المحلية معروضة كنسخة احتياطية.

## 3) الفروق بين الخطة "التجريبية" والخطة "الحقيقية"

الواجهة الأصلية بُنيت على بيانات وهمية ثابتة (PLAN/STEPS) بخطوات لها
bullets وروابط جهات حكومية (GOV) وتبعيات (deps) محسوبة يدويًا. أما مخرجات
الوكيل الفعلية (`plan.plan` من `/api/plan`) فشكلها:

```json
{"order": 1, "title": "...", "authority": "...",
 "description": "...", "status": "verified | unverified | blocked",
 "caveat": "... أو null"}
```

أي: **لا توجد روابط جهات جاهزة للنقر، ولا bullets تفصيلية، ولا تبعيات
كأرقام خطوات** — الوكيل يرتّب الخطوات منطقيًا عبر `order` فقط، ويصف حالة كل
خطوة (`status`) بدل قائمة نقاط. عدّلتُ `buildPlan()` لتعرض هذا بأفضل شكل
ممكن ضمن نفس تصميم الواجهة (تحويل status إلى ملاحظة تنبيه، وإلحاق اسم
الجهة بنص الوصف)، لكن إن أردتِ الشكل الأصلي بالضبط (bullets + روابط
حكومية قابلة للنقر) فالخيار الأنظف هو توسيع مخطط JSON الذي تُعيده دالة
`planner_agent_v2` في `nawah_pipeline.py` ليشمل `bullets` و`source_url`
لكل خطوة، بدل التوفيق بعد الاستلام. هذا تعديل على "عقل" الوكيل نفسه
(ملف `nawah_pipeline.py`، بالضبط كما لو عدّلتِه في النوتبوك) — لم ألمسه
لأن السؤال كان عن *الربط* لا عن تغيير سلوك الوكيل، لكنه يستحق فقرة قصيرة
لأنه القرار الطبيعي التالي.

## 4) النشر (بعيدًا عن Colab)

الخادم عادي تمامًا (FastAPI/uvicorn) — لا يحتاج Colab أو ngrok. أي مضيف
يشغّل حاويات بايثون يناسبه، مثل Render أو Railway أو Fly.io أو خادمكم
الخاص:

```bash
docker build -t nawah-api .
docker run -p 8000:8000 --env-file .env nawah-api
```

بعد النشر:
1. غيّري `const API_BASE = "http://localhost:8000";` في `nawah_connected.html`
   إلى الرابط العام لخادمك (مثل `https://nawah-api.onrender.com`).
2. في `server.py`، قيّدي `allow_origins=["*"]` إلى دومين موقعك الفعلي بدل
   `*` (السطر معلَّم بـ `TODO` في الملف).
3. ارفعي `nawah_connected.html` على أي استضافة ثابتة (Netlify, Vercel,
   GitHub Pages، أو خادمكم نفسه).

## 5) ملاحظة مهمة عن أسماء النماذج

كما في تنبيه النوتبوك الأصلي: أسماء النماذج (`gpt-5.6-sol` وأخواتها) وأداة
`web_search` هي أفضل تخمين وقت كتابة النوتبوك — تأكدي منها في توثيق
OpenAI الرسمي قبل التشغيل الفعلي، فقد تحتاجين لتحديثها في أعلى
`nawah_pipeline.py` (`GPT_SOL` / `GPT_TERRA` / `GPT_LUNA`).


## python -m http.server 5500
## python -m uvicorn server:app --reload --port 8000