import random
from datetime import datetime, timedelta
from sqlite_logger import SQLiteLogger

def main():
    print("Generating mock query logs...")
    logger = SQLiteLogger()
    
    # Clean existing logs
    with logger.conn:
        logger.conn.execute("DELETE FROM ratings")
        logger.conn.execute("DELETE FROM query_logs")
        
    models_pool = [
        ("hyundai", "creta"),
        ("hyundai", "alcazar"),
        ("hyundai", "i20"),
        ("hyundai", "verna"),
        ("maruti-suzuki", "swift"),
        ("maruti-suzuki", "baleno"),
        ("maruti-suzuki", "brezza"),
        ("maruti-suzuki", "grand-vitara"),
        ("tata", "nexon"),
        ("tata", "punch"),
        ("tata", "harrier"),
        ("tata", "safari"),
        ("tata", "tiago")
    ]
    
    queries = [
        "Does the Creta have a sunroof?",
        "What is the boot space of the Tata Nexon?",
        "Compare Swift and Creta boot space.",
        "Wheelbase of Hyundai Creta",
        "Does Tata Harrier have ADAS?",
        "Is there a CNG variant for Brezza?",
        "What is the mileage of Tata Safari?",
        "Swift turning radius",
        "Creta fuel tank capacity",
        "Bigger engine Nexon or Swift?",
        "Altroz ground clearance",
        "Is Nexon EV mileage good?",
        "Show spec list of Grand Vitara",
        "Does Tiago have side airbags?"
    ]
    
    confidences = ["high", "high", "medium", "low", "not_found"]
    
    # Generate logs for past 30 days
    now = datetime.utcnow()
    total_logs = 0
    
    for day in range(30):
        log_date = now - timedelta(days=(30 - day))
        # Random number of queries per day (3 to 8)
        num_queries = random.randint(3, 8)
        for _ in range(num_queries):
            brand, model = random.choice(models_pool)
            query = random.choice(queries)
            is_comp = "compare" in query.lower() or "or" in query.lower()
            confidence = random.choice(confidences)
            latency = random.uniform(300, 2500) if confidence != "not_found" else random.uniform(100, 800)
            tokens = random.randint(200, 1500)
            
            # Format custom date
            # Add random hour/minute/second
            time_offset = timedelta(
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )
            timestamp = (log_date.replace(hour=0, minute=0, second=0) + time_offset).isoformat()
            
            with logger.conn:
                logger.conn.execute("""
                    INSERT INTO query_logs (
                        timestamp, query, rewritten_query, brand, model, is_comparison,
                        confidence, latency_ms, tokens_used, response_json, error_msg
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """, (
                    timestamp,
                    query,
                    query,
                    brand,
                    model,
                    1 if is_comp else 0,
                    confidence,
                    latency,
                    tokens,
                    '{"answer": "Mocked response content"}'
                ))
            total_logs += 1
            
    print(f"Pre-populated database with {total_logs} mock logs successfully.")

if __name__ == "__main__":
    main()
