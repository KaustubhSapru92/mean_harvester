from math_generator.workflow.config_loader import ConfigLoader
from math_generator.pipelines.data_pipeline import run_data_pipeline
from math_generator.pipelines.vwap_pipeline import run_vwap_pipeline

config = ConfigLoader("math_generator/config.yml")
config.config["market_data"]["lookback_days"] = 59
config.config["validation"] = {
    "require_convergence": False,
    "allow_smoke_fallback": True,
}

clean_df, _ = run_data_pipeline(config)

vwap_windows = [100, 200, 300]

result = run_vwap_pipeline(
    clean_df=clean_df,
    vwap_windows=vwap_windows,
    config=config,
    roll_len=50,
)

print("Validation config:", config.config["validation"])
print("require_convergence:", config.get("validation", "require_convergence"))
print("allow_smoke_fallback:", config.get("validation", "allow_smoke_fallback"))

print("\nSMOKE TEST COMPLETE")
print("Frozen params:", result["frozen_params"])
print("Summary table:")
print(result["summary_table"])
print("Degradation table:")
print(result["degradation_table"])
print("Cost impact table:")
print(result["cost_impact_table"])
