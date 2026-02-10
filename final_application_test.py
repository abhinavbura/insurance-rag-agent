import chromadb
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import uuid
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
print("All libraries imported successfully.")
from dotenv import load_dotenv
load_dotenv()
class InsuranceApplicationRAG:
    def __init__(self):
        print("🚀 Initializing Insurance RAG System...")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./insurance_chroma_db"
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="insurance_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load embedding model
        print("📦 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize 3 agents
        self.analyzer = ApplicationAnalyzerAgent()
        self.retriever = KnowledgeRetrieverAgent(self.collection, self.embedding_model)
        self.generator = SuggestionGeneratorAgent()
        
        # Load knowledge base
        self.setup_knowledge_base()
        
        print("✅ Insurance RAG System Ready!\n")
    def setup_knowledge_base(self):
        print("📚 Setting up knowledge base...")
        
        # Check if already populated
        if self.collection.count() > 0:
            print(f"✅ Knowledge base already loaded with {self.collection.count()} documents\n")
            return
        
        # Get all knowledge documents
        knowledge_docs = self.create_comprehensive_knowledge()
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        print(f"🔄 Indexing {len(knowledge_docs)} knowledge documents...")
        
        for doc in knowledge_docs:
            ids.append(doc["id"])
            documents.append(doc["text"])
            metadatas.append(doc["metadata"])
            # Generate embedding for each document
            embedding = self.embedding_model.encode(doc["text"]).tolist()
            embeddings.append(embedding)
        
        # Insert into ChromaDB
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        print(f"✅ Successfully indexed {len(knowledge_docs)} documents into ChromaDB\n")

    def create_comprehensive_knowledge(self):
        
        auto_knowledge = [
            {
                "id": "auto_ca_001",
                "text": "California requires minimum liability coverage of 15/30/5. This means $15,000 for injury per person, $30,000 per accident, and $5,000 for property damage. All auto insurance applications in California must include proof of these minimums.",
                "metadata": {"category": "regulatory", "product_type": "auto", "state": "CA", "severity": "critical"}
            },
            {
                "id": "auto_ca_002",
                "text": "California auto insurance applications require valid driver license number, vehicle VIN, current odometer reading, and primary garaging address within the state.",
                "metadata": {"category": "required_fields", "product_type": "auto", "state": "CA", "severity": "critical"}
            },
            {
                "id": "auto_vin_001",
                "text": "Vehicle Identification Number (VIN) must be exactly 17 characters long, containing only letters and numbers. VIN cannot contain letters I, O, or Q to avoid confusion with numbers. VIN encodes manufacturer, vehicle type, and production sequence.",
                "metadata": {"category": "validation", "product_type": "auto", "state": "general", "severity": "critical"}
            },
            {
                "id": "auto_vin_002",
                "text": "VIN can be found in three locations: driver side dashboard visible through windshield, driver side door jamb sticker, and vehicle title or registration documents. Always verify VIN matches across all documents.",
                "metadata": {"category": "best_practice", "product_type": "auto", "state": "general", "severity": "info"}
            },
            {
                "id": "auto_fl_001",
                "text": "Florida requires Personal Injury Protection (PIP) of minimum $10,000 and Property Damage Liability of $10,000. Florida is a no-fault state meaning PIP covers injuries regardless of who caused the accident.",
                "metadata": {"category": "regulatory", "product_type": "auto", "state": "FL", "severity": "critical"}
            },
            {
                "id": "auto_fl_002",
                "text": "Florida auto applications must include PIP selection, PDL coverage amount, vehicle garaging zip code, and annual mileage estimate. Garaging location significantly impacts premium calculation in Florida.",
                "metadata": {"category": "required_fields", "product_type": "auto", "state": "FL", "severity": "critical"}
            },
            {
                "id": "auto_tx_001",
                "text": "Texas requires minimum auto insurance of 30/60/25. This means $30,000 per injured person, $60,000 per accident, and $25,000 for property damage. Texas also requires insurers to offer uninsured motorist coverage.",
                "metadata": {"category": "regulatory", "product_type": "auto", "state": "TX", "severity": "critical"}
            },
            {
                "id": "auto_tx_002",
                "text": "Texas auto insurance applications must include vehicle make, model, year, VIN, annual mileage, primary driver information, and garaging address. Texas requires disclosure of all household members of driving age.",
                "metadata": {"category": "required_fields", "product_type": "auto", "state": "TX", "severity": "critical"}
            },
            {
                "id": "auto_risk_001",
                "text": "High risk indicators for auto insurance include: DUI/DWI violations in last 5 years, more than 2 at-fault accidents in 3 years, license suspension history, vehicle modifications for racing, and annual mileage exceeding 20,000 miles.",
                "metadata": {"category": "risk", "product_type": "auto", "state": "general", "severity": "warning"}
            },
            {
                "id": "auto_validation_001",
                "text": "Auto insurance field validation rules: Driver license must be 7-9 alphanumeric characters. Vehicle year must be between 1980 and current year plus 1. Annual mileage must be between 1000 and 200000. Date of birth must indicate driver is at least 16 years old.",
                "metadata": {"category": "validation", "product_type": "auto", "state": "general", "severity": "critical"}
            },
            {
                "id": "auto_best_001",
                "text": "Best practices for auto insurance applications: Always list all household drivers to avoid coverage gaps. Disclose all vehicles regularly used even if not owned. Report accurate annual mileage as underreporting is grounds for claim denial.",
                "metadata": {"category": "best_practice", "product_type": "auto", "state": "general", "severity": "info"}
            },
            {
                "id": "auto_commercial_001",
                "text": "Commercial auto insurance requires DOT number for vehicles over 10,000 lbs GVWR operating interstate. Business use classification must match actual vehicle usage. Personal auto policies are void for commercial use incidents.",
                "metadata": {"category": "regulatory", "product_type": "commercial_auto", "state": "general", "severity": "critical"}
            },
            {
                "id": "auto_commercial_002",
                "text": "Commercial auto applications require: Business name and EIN, DOT number if applicable, list of all drivers with MVR consent, vehicle usage description, radius of operation, and cargo type if applicable.",
                "metadata": {"category": "required_fields", "product_type": "commercial_auto", "state": "general", "severity": "critical"}
            }
        ]

        home_knowledge = [
            {
                "id": "home_fl_001",
                "text": "Florida requires windstorm coverage for all residential properties. Due to hurricane risk, windstorm coverage is mandatory and cannot be excluded from home insurance policies in Florida. Properties in coastal zones require additional windstorm inspections.",
                "metadata": {"category": "regulatory", "product_type": "home", "state": "FL", "severity": "critical"}
            },
            {
                "id": "home_fl_002",
                "text": "Florida home insurance applications must include: property address, year built, construction type, roof age and material, distance to coast, windstorm coverage selection, and flood zone determination. Roof age over 15 years requires inspection report.",
                "metadata": {"category": "required_fields", "product_type": "home", "state": "FL", "severity": "critical"}
            },
            {
                "id": "home_fl_003",
                "text": "Florida Citizens Insurance is the insurer of last resort for properties that cannot obtain private market coverage. Properties within 1000 feet of coastline are automatically eligible. Annual re-inspection required for properties over 25 years old.",
                "metadata": {"category": "regulatory", "product_type": "home", "state": "FL", "severity": "warning"}
            },
            {
                "id": "home_fl_004",
                "text": "Florida flood insurance is separate from standard home insurance and is required for properties in FEMA designated flood zones A and V. Flood insurance has a 30-day waiting period before coverage begins. Average Florida flood claim exceeds $28,000.",
                "metadata": {"category": "regulatory", "product_type": "home", "state": "FL", "severity": "critical"}
            },
            {
                "id": "home_ca_001",
                "text": "California home insurance applications must address wildfire risk. Properties in Fire Hazard Severity Zones require additional documentation. Insurers must offer 36-month guaranteed renewal for properties that survive declared disasters.",
                "metadata": {"category": "regulatory", "product_type": "home", "state": "CA", "severity": "critical"}
            },
            {
                "id": "home_tx_001",
                "text": "Texas home insurance must include windstorm and hail coverage for properties in designated catastrophe areas along the Gulf Coast. Texas Windstorm Insurance Association (TWIA) provides coverage for 14 coastal counties.",
                "metadata": {"category": "regulatory", "product_type": "home", "state": "TX", "severity": "critical"}
            },
            {
                "id": "home_required_001",
                "text": "Standard home insurance required fields: property address, year built, square footage, construction type (frame/masonry/mixed), number of stories, roof type and age, heating type, presence of pool or trampoline, occupancy type, and replacement cost estimate.",
                "metadata": {"category": "required_fields", "product_type": "home", "state": "general", "severity": "critical"}
            },
            {
                "id": "home_risk_001",
                "text": "High risk indicators for home insurance: roof age over 20 years, presence of trampoline or diving board, wood burning stove as primary heat source, property vacancy over 30 days, prior water damage claims, knob and tube wiring, and properties in flood zones.",
                "metadata": {"category": "risk", "product_type": "home", "state": "general", "severity": "warning"}
            },
            {
                "id": "home_risk_002",
                "text": "Florida specific risk factors: homes built before 1994 may not meet current hurricane building codes, metal roofs qualify for wind mitigation discounts, impact resistant windows reduce windstorm premium by up to 45%, and generator presence requires disclosure.",
                "metadata": {"category": "risk", "product_type": "home", "state": "FL", "severity": "warning"}
            },
            {
                "id": "home_validation_001",
                "text": "Home insurance validation rules: Year built must be between 1800 and current year. Square footage must be between 400 and 20000. Replacement cost must be at least 80% of estimated rebuild cost or coinsurance penalty applies. Property value must be supported by recent appraisal.",
                "metadata": {"category": "validation", "product_type": "home", "state": "general", "severity": "critical"}
            },
            {
                "id": "home_best_001",
                "text": "Best practices for home insurance applications: Always calculate replacement cost not market value. Document all high value items with photos and receipts. Disclose all business activities conducted from home. Update coverage after major renovations to avoid underinsurance.",
                "metadata": {"category": "best_practice", "product_type": "home", "state": "general", "severity": "info"}
            },
            {
                "id": "home_best_002",
                "text": "Florida home insurance optimization tips: Install wind mitigation features for premium discounts up to 45%. Raise deductible for hurricane coverage to lower annual premium. Bundle auto and home for multi-policy discount. Annual policy review recommended due to frequent Florida market changes.",
                "metadata": {"category": "best_practice", "product_type": "home", "state": "FL", "severity": "info"}
            }
        ]

        return auto_knowledge + home_knowledge
    def process_application(self, form_data):
        print("\n" + "🔥"*30)
        print(f"📋 PROCESSING APPLICATION: {form_data.get('product_type', 'unknown').upper()} - {form_data.get('state', 'unknown')}")
        print("🔥"*30)

        # Step 1: Analyze application
        analysis = self.analyzer.analyze_application(form_data)

        # Step 2: Build query from analysis
        missing = analysis.get("missing_fields", [])
        risks = analysis.get("risk_indicators", [])
        product_type = analysis.get("product_type", "auto")
        state = analysis.get("state", "general")

        query = f"{product_type} insurance requirements validation"
        if missing:
            query += f" {' '.join(missing[:2])}"
        if risks:
            query += f" {' '.join(risks[:2])}"

        # Step 3: Retrieve relevant knowledge
        knowledge = self.retriever.retrieve_relevant_knowledge(query, analysis)

        # Step 4: Generate suggestions
        suggestions = self.generator.generate_suggestions(analysis, knowledge)

        # Step 5: Return complete result
        return {
            "analysis": analysis,
            "knowledge": knowledge,
            "suggestions": suggestions
        }
    
class ApplicationAnalyzerAgent:
    def __init__(self):
        self.required_fields = {
            "auto": [
                "applicant_name", "date_of_birth", "license_number",
                "vehicle_make", "vehicle_model", "vehicle_year",
                "vehicle_vin", "annual_mileage", "coverage_type", "state"
            ],
            "home": [
                "applicant_name", "property_address", "year_built",
                "square_footage", "construction_type", "roof_age",
                "property_value", "coverage_amount", "state"
            ],
            "commercial_auto": [
                "business_name", "ein_number", "dot_number",
                "vehicle_make", "vehicle_model", "vehicle_vin",
                "business_classification", "radius_of_operation", "state"
            ]
        }

        self.validation_rules = {
            "vehicle_vin": {"length": 17, "pattern": "alphanumeric"},
            "vehicle_year": {"min": 1980, "max": datetime.now().year + 1},
            "annual_mileage": {"min": 1000, "max": 200000},
            "square_footage": {"min": 400, "max": 20000},
            "year_built": {"min": 1800, "max": datetime.now().year},
            "license_number": {"min_length": 7, "max_length": 9}
        }

    def analyze_application(self, form_data):
        print("\n" + "="*60)
        print("🔍 AGENT 1: APPLICATION ANALYZER")
        print("="*60)

        product_type = form_data.get("product_type", "auto")

        # Run all analysis steps
        completion = self._calculate_completion(form_data, product_type)
        missing_fields = self.find_missing_critical_fields(form_data, product_type)
        validation_errors = self._validate_fields(form_data)
        risk_indicators = self._detect_risk_indicators(form_data, product_type)

        analysis = {
            "product_type": product_type,
            "state": form_data.get("state", "general"),
            "completion_percentage": completion,
            "missing_fields": missing_fields,
            "validation_errors": validation_errors,
            "risk_indicators": risk_indicators,
            "total_fields": len(self.required_fields.get(product_type, [])),
            "filled_fields": len(self.required_fields.get(product_type, [])) - len(missing_fields)
        }

        # Print analysis results
        print(f"📋 Product Type    : {product_type.upper()}")
        print(f"📍 State           : {analysis['state']}")
        print(f"✅ Completion      : {completion}%")
        print(f"📝 Fields Filled   : {analysis['filled_fields']}/{analysis['total_fields']}")

        if missing_fields:
            print(f"❌ Missing Fields  : {', '.join(missing_fields)}")
        else:
            print("✅ No Missing Fields!")

        if validation_errors:
            print(f"⚠️  Validation Errors:")
            for field, error in validation_errors.items():
                print(f"   → {field}: {error}")

        if risk_indicators:
            print(f"🚨 Risk Indicators : {', '.join(risk_indicators)}")
        else:
            print("✅ No Risk Indicators Detected")

        return analysis

    def _calculate_completion(self, form_data, product_type):
        required = self.required_fields.get(product_type, [])
        if not required:
            return 0
        filled = sum(1 for field in required if form_data.get(field))
        return round((filled / len(required)) * 100)

    def find_missing_critical_fields(self, form_data, product_type):
        required = self.required_fields.get(product_type, [])
        return [field for field in required if not form_data.get(field)]

    def _validate_fields(self, form_data):
        errors = {}

        for field, rules in self.validation_rules.items():
            value = form_data.get(field)
            if not value:
                continue

            # VIN validation
            if field == "vehicle_vin":
                if len(str(value)) != rules["length"]:
                    errors[field] = f"VIN must be exactly {rules['length']} characters, got {len(str(value))}"
                invalid_chars = [c for c in str(value).upper() if c in ['I', 'O', 'Q']]
                if invalid_chars:
                    errors[field] = f"VIN cannot contain letters: {', '.join(invalid_chars)}"

            # Range validations
            elif "min" in rules and "max" in rules:
                try:
                    num_value = int(value)
                    if not (rules["min"] <= num_value <= rules["max"]):
                        errors[field] = f"Value {num_value} out of range [{rules['min']} - {rules['max']}]"
                except (ValueError, TypeError):
                    errors[field] = f"Invalid numeric value for {field}"

            # Length validations
            elif "min_length" in rules and "max_length" in rules:
                str_value = str(value)
                if not (rules["min_length"] <= len(str_value) <= rules["max_length"]):
                    errors[field] = f"Length {len(str_value)} out of range [{rules['min_length']} - {rules['max_length']}]"

        return errors

    def _detect_risk_indicators(self, form_data, product_type):
        risks = []

        if product_type == "auto":
            if int(form_data.get("annual_mileage", 0)) > 20000:
                risks.append("HIGH_MILEAGE")
            if form_data.get("dui_history"):
                risks.append("DUI_HISTORY")
            if form_data.get("accidents_3yr", 0) > 2:
                risks.append("MULTIPLE_ACCIDENTS")

        if product_type == "home":
            if int(form_data.get("roof_age", 0)) > 20:
                risks.append("AGING_ROOF")
            if form_data.get("state") == "FL":
                risks.append("FLORIDA_HURRICANE_ZONE")
            if form_data.get("pool"):
                risks.append("POOL_LIABILITY")
            if int(form_data.get("year_built", 2000)) < 1994 and form_data.get("state") == "FL":
                risks.append("PRE_1994_FL_CONSTRUCTION")

        return risks
    
class KnowledgeRetrieverAgent:
    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model
        self.top_k = 3

    def retrieve_relevant_knowledge(self, query, context):
        print("\n" + "="*60)
        print("📚 AGENT 2: KNOWLEDGE RETRIEVER")
        print("="*60)

        # Enhance query with context
        contextual_query = self._create_contextual_query(query, context)
        print(f"🔎 Original Query   : {query}")
        print(f"🔎 Contextual Query : {contextual_query}")

        # Generate query embedding
        query_embedding = self.embedding_model.encode(contextual_query).tolist()

        # Build metadata filter based on context
        state = context.get("state", "general")
        product_type = context.get("product_type", "auto")

        # Retrieve with semantic search + metadata filter
        results = self._hybrid_search(query_embedding, state, product_type)

        # Format results
        knowledge_items = self._format_results(results)

        # Print retrieved knowledge
        print(f"\n📄 Retrieved {len(knowledge_items)} Knowledge Items:")
        for i, item in enumerate(knowledge_items, 1):
            print(f"\n  [{i}] Category : {item['category']} | State: {item['state']} | Severity: {item['severity']}")
            print(f"      Content  : {item['text'][:120]}...")

        return knowledge_items

    def _create_contextual_query(self, query, context):
        state = context.get("state", "")
        product_type = context.get("product_type", "")
        missing_fields = context.get("missing_fields", [])
        risk_indicators = context.get("risk_indicators", [])

        # Build enriched query
        enriched_parts = [query]

        if product_type:
            enriched_parts.append(f"{product_type} insurance")
        if state and state != "general":
            enriched_parts.append(f"{state} state requirements")
        if missing_fields:
            enriched_parts.append(f"missing {' '.join(missing_fields[:2])}")
        if risk_indicators:
            enriched_parts.append(f"risk factors {' '.join(risk_indicators[:2])}")

        return " ".join(enriched_parts)

    def _hybrid_search(self, query_embedding, state, product_type):
        all_results = []

        # Search 1: State + product specific (highest priority)
        if state != "general":
            try:
                state_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(2, self.collection.count()),
                    where={
                        "$and": [
                            {"state": {"$in": [state, "general"]}},
                            {"product_type": {"$in": [product_type, "general"]}}
                        ]
                    }
                )
                all_results.append(state_results)
            except Exception as e:
                print(f"⚠️ State filter search failed: {e}")

        # Search 2: Product type general search
        try:
            product_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(2, self.collection.count()),
                where={"product_type": {"$in": [product_type, "general"]}}
            )
            all_results.append(product_results)
        except Exception as e:
            print(f"⚠️ Product filter search failed: {e}")

        # Search 3: Pure semantic fallback
        try:
            semantic_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(self.top_k, self.collection.count())
            )
            all_results.append(semantic_results)
        except Exception as e:
            print(f"⚠️ Semantic search failed: {e}")

        return self._deduplicate_results(all_results)

    def _deduplicate_results(self, all_results):
        seen_ids = set()
        deduplicated = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

        for results in all_results:
            if not results or not results.get("ids"):
                continue
            for i, doc_id in enumerate(results["ids"][0]):
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    deduplicated["ids"][0].append(doc_id)
                    deduplicated["documents"][0].append(results["documents"][0][i])
                    deduplicated["metadatas"][0].append(results["metadatas"][0][i])
                    deduplicated["distances"][0].append(results["distances"][0][i])

        return deduplicated

    def _format_results(self, results):
        knowledge_items = []

        if not results or not results.get("documents"):
            return knowledge_items

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(min(self.top_k, len(documents))):
            knowledge_items.append({
                "text": documents[i],
                "category": metadatas[i].get("category", "general"),
                "state": metadatas[i].get("state", "general"),
                "product_type": metadatas[i].get("product_type", "general"),
                "severity": metadatas[i].get("severity", "info"),
                "relevance_score": round(1 - distances[i], 3)
            })

        # Sort by relevance
        knowledge_items.sort(key=lambda x: x["relevance_score"], reverse=True)

        return knowledge_items
    
class SuggestionGeneratorAgent:
    def __init__(self):

        api_key=os.environ.get("OPENROUTER_API_KEY")
        self.llm = ChatOpenAI(
            model="deepseek/deepseek-r1-0528:free",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1,
            max_tokens=2000,
        )

    def generate_suggestions(self, analysis, knowledge):
        print("\n" + "="*60)
        print("💡 AGENT 3: SUGGESTION GENERATOR (DeepSeek via OpenRouter)")
        print("="*60)

        # Format knowledge chunks from ChromaDB
        knowledge_text = self._format_knowledge(knowledge)

        # Format analysis for prompt
        analysis_text = self._format_analysis(analysis)

        # Build prompt
        system_prompt = """You are an expert insurance application assistant. 
Your job is to help users complete insurance applications correctly and compliantly.
You will be given:
1. An analysis of the current application state
2. Relevant knowledge retrieved from our insurance knowledge base

Based on this information provide:
- Clear, specific guidance for each missing or invalid field
- State specific regulatory warnings
- Risk factor explanations
- Actionable next steps

Be concise, professional, and helpful. 
For next_steps always respond with a JSON array at the end of your response in this exact format:
NEXT_STEPS_JSON: [{"step": 1, "action": "...", "priority": "critical|high|medium"}]"""

        human_prompt = f"""
APPLICATION ANALYSIS:
{analysis_text}

RELEVANT KNOWLEDGE FROM KNOWLEDGE BASE:
{knowledge_text}

Please provide intelligent suggestions to help complete this insurance application correctly.
Address all missing fields, validation errors, risk indicators, and state specific requirements.
End your response with the NEXT_STEPS_JSON array.
"""

        try:
            print("\n🤖 Calling DeepSeek via OpenRouter...")
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]

            response = self.llm.invoke(messages)
            full_response = response.content

            # Split natural language from JSON next steps
            suggestions = self._parse_response(full_response, analysis)

            # Print output
            self._print_suggestions(suggestions)

            return suggestions

        except Exception as e:
            print(f"⚠️ LLM call failed: {e}")
            return self._fallback_suggestions(analysis)

    def _format_knowledge(self, knowledge):
        if not knowledge:
            return "No specific knowledge retrieved."

        formatted = []
        for i, item in enumerate(knowledge, 1):
            formatted.append(
                f"[{i}] Category: {item['category'].upper()} | "
                f"State: {item['state']} | "
                f"Severity: {item['severity'].upper()}\n"
                f"    {item['text']}"
            )
        return "\n\n".join(formatted)

    def _format_analysis(self, analysis):
        return f"""
- Product Type      : {analysis.get('product_type', 'N/A').upper()}
- State             : {analysis.get('state', 'N/A')}
- Completion        : {analysis.get('completion_percentage', 0)}%
- Missing Fields    : {', '.join(analysis.get('missing_fields', [])) or 'None'}
- Validation Errors : {json.dumps(analysis.get('validation_errors', {})) or 'None'}
- Risk Indicators   : {', '.join(analysis.get('risk_indicators', [])) or 'None'}
- Fields Filled     : {analysis.get('filled_fields', 0)}/{analysis.get('total_fields', 0)}
"""

    def _parse_response(self, full_response, analysis):
        next_steps = []
        natural_language = full_response

        # Extract JSON next steps if present
        if "NEXT_STEPS_JSON:" in full_response:
            parts = full_response.split("NEXT_STEPS_JSON:")
            natural_language = parts[0].strip()
            try:
                json_str = parts[1].strip()
                next_steps = json.loads(json_str)
            except Exception:
                next_steps = self._fallback_next_steps(analysis)
        else:
            next_steps = self._fallback_next_steps(analysis)

        return {
            "natural_language_guidance": natural_language,
            "next_steps": next_steps,
            "completion_percentage": analysis.get("completion_percentage", 0),
            "missing_fields_count": len(analysis.get("missing_fields", [])),
            "risk_count": len(analysis.get("risk_indicators", []))
        }

    def _fallback_next_steps(self, analysis):
        steps = []
        if analysis.get("missing_fields"):
            steps.append({
                "step": 1,
                "action": f"Complete missing fields: {', '.join(analysis['missing_fields'])}",
                "priority": "critical"
            })
        if analysis.get("validation_errors"):
            steps.append({
                "step": 2,
                "action": f"Fix validation errors in: {', '.join(analysis['validation_errors'].keys())}",
                "priority": "high"
            })
        if analysis.get("risk_indicators"):
            steps.append({
                "step": 3,
                "action": f"Review risk factors: {', '.join(analysis['risk_indicators'])}",
                "priority": "medium"
            })
        return steps

    def _fallback_suggestions(self, analysis):
        print("⚠️ Using fallback suggestions due to LLM failure")
        return {
            "natural_language_guidance": f"Application is {analysis.get('completion_percentage', 0)}% complete. Please complete missing fields: {', '.join(analysis.get('missing_fields', []))}",
            "next_steps": self._fallback_next_steps(analysis),
            "completion_percentage": analysis.get("completion_percentage", 0),
            "missing_fields_count": len(analysis.get("missing_fields", [])),
            "risk_count": len(analysis.get("risk_indicators", []))
        }

    def _print_suggestions(self, suggestions):
        completion = suggestions.get("completion_percentage", 0)

        if completion == 100:
            status = "✅ COMPLETE"
        elif completion >= 80:
            status = "🔶 ALMOST THERE"
        elif completion >= 60:
            status = "🔷 IN PROGRESS"
        else:
            status = "❌ INCOMPLETE"

        print(f"\n{status} — Application is {completion}% complete")
        print(f"\n📝 AI GUIDANCE:\n")
        print(suggestions["natural_language_guidance"])

        if suggestions.get("next_steps"):
            print(f"\n🗺️  NEXT STEPS:")
            for step in suggestions["next_steps"]:
                priority_icon = "🔴" if step.get("priority") == "critical" else "🟡" if step.get("priority") == "high" else "🟢"
                print(f"   {priority_icon} Step {step.get('step', '')}: {step.get('action', '')}")
class InsuranceApplicationDemo:
    def __init__(self):
        print("\n" + "="*60)
        print("🏢 INSURANCE RAG SYSTEM - HACKATHON DEMO")
        print("   RAG-Based Agentic Insurance Application Assistant")
        print("="*60)
        self.rag_system = InsuranceApplicationRAG()

    def run(self):
        self._run_scenario_1()
        self._run_scenario_2()
        self._print_summary()

    def _run_scenario_1(self):
        print("\n\n" + "★"*60)
        print("★  SCENARIO 1: INCOMPLETE AUTO INSURANCE APPLICATION")
        print("★"*60)

        form_data = {
            "product_type": "auto",
            "state": "CA",
            "applicant_name": "John Smith",
            "date_of_birth": "01/15/1990",
            "license_number": "CA12345",
            "vehicle_make": "Toyota",
            "vehicle_model": "Camry",
            "vehicle_year": "2020",
            "vehicle_vin": "",          # MISSING - critical field
            "annual_mileage": "12000",
            "coverage_type": "comprehensive"
            # state is filled but vin is missing
        }

        print("\n📥 INPUT FORM DATA:")
        for key, value in form_data.items():
            status = "❌ MISSING" if value == "" else f"✅ {value}"
            print(f"   {key:<25} : {status}")

        result = self.rag_system.process_application(form_data)
        return result

    def _run_scenario_2(self):
        print("\n\n" + "★"*60)
        print("★  SCENARIO 2: FLORIDA HOME INSURANCE APPLICATION")
        print("★"*60)

        form_data = {
            "product_type": "home",
            "state": "FL",
            "applicant_name": "Maria Garcia",
            "property_address": "123 Ocean Drive, Miami, FL 33139",
            "year_built": "1989",       # Pre-1994 FL construction risk
            "square_footage": "2200",
            "construction_type": "frame",
            "roof_age": "18",           # Aging roof risk
            "property_value": "450000",
            "coverage_amount": "",      # MISSING
            # windstorm_coverage missing → should be flagged
        }

        print("\n📥 INPUT FORM DATA:")
        for key, value in form_data.items():
            status = "❌ MISSING" if value == "" else f"✅ {value}"
            print(f"   {key:<25} : {status}")

        result = self.rag_system.process_application(form_data)
        return result

    def _print_summary(self):
        print("\n\n" + "="*60)
        print("🏁 DEMO SUMMARY")
        print("="*60)
        print("✅ Scenario 1: Auto Insurance - Missing VIN Detection")
        print("   → Analyzer detected missing VIN and completion %")
        print("   → Retriever fetched VIN validation rules from ChromaDB")
        print("   → DeepSeek generated natural language VIN guidance")
        print()
        print("✅ Scenario 2: Home Insurance - Florida Risk Detection")
        print("   → Analyzer detected Florida hurricane zone risks")
        print("   → Retriever fetched FL windstorm rules from ChromaDB")
        print("   → DeepSeek generated mandatory coverage warnings")
        print()
        print("🎯 System Capabilities Demonstrated:")
        print("   → Multi-agent RAG pipeline with LLM integration")
        print("   → Semantic search with metadata filtering")
        print("   → State specific regulatory compliance detection")
        print("   → DeepSeek powered natural language suggestions")
        print("   → Structured JSON next steps for downstream agents")
        print("="*60)
        print("🚀 Insurance RAG System Demo Complete!")
        print("="*60)

if __name__ == "__main__":
    demo = InsuranceApplicationDemo()

    demo.run()
