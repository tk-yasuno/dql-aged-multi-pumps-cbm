import yaml

try:
    with open('config_pump_cbm_v047.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("✅ Config loaded successfully")
    print(f"Equipment count: {len(config['multi_equipment']['target_equipment_list'])}")
    
    for i, eq in enumerate(config['multi_equipment']['target_equipment_list']):
        print(f"  {i+1}. {eq['equipment_name']} (ID: {eq['equipment_id']})")
        
except Exception as e:
    print(f"❌ Error: {e}")