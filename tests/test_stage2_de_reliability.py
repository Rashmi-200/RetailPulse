import pandas as pd
import pytest
from deltalake import DeltaTable
from sqlalchemy import text

from src.contracts.contract_engine import DataContractEngine
from src.database.db_connection import get_engine
from src.lakehouse.gold_builder import GoldStarSchemaBuilder
from src.lineage.lineage_tracker import OpenLineageTracker
from src.observability.tracker import ObservabilityTracker
from src.schema_evolution.registry import SchemaEvolutionRegistry
from src.utils.config_loader import load_yaml_config


def test_schema_evolution_all_five_scenarios():
    engine = get_engine()
    tracker = ObservabilityTracker(engine)
    registry = SchemaEvolutionRegistry(engine, tracker)
    contracts_cfg = load_yaml_config("config/data_contracts.yaml")
    contract_engine = DataContractEngine(engine, contracts_cfg)

    ds_name = "pytest_schema_evolution_probe"
    # Clean any previous test runs for this probe dataset
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM meta.schema_versions WHERE dataset_name = :ds"), {"ds": ds_name})
        conn.execute(text("DELETE FROM meta.incidents WHERE dataset_name = :ds"), {"ds": ds_name})

    base_df = pd.DataFrame({"order_id": ["o1", "o2"], "amount": [10.0, 20.0]})
    v1 = registry.check_and_register(base_df, ds_name, "bronze", required_columns=["order_id"])
    assert v1.version_number == 1 and v1.change_type == "initial"

    # Scenario 5: Column reordering -> Canonical SHA-256 hash must NOT trigger false drift
    reordered_df = base_df[["amount", "order_id"]]
    v_reorder = registry.check_and_register(reordered_df, ds_name, "bronze", required_columns=["order_id"])
    assert v_reorder.version_number == 1 and v_reorder.change_type == "unchanged"

    # Scenario 1: Nullable column addition -> Non-breaking evolution (v2)
    added_col_df = base_df.copy()
    added_col_df["optional_note"] = [None, "promo"]
    v2 = registry.check_and_register(added_col_df, ds_name, "bronze", required_columns=["order_id"])
    assert v2.version_number == 2 and v2.is_breaking is False and v2.change_type == "non_breaking_evolution"

    # Scenario 2: Column removal -> Breaking drift (must raise ValueError)
    removed_col_df = added_col_df[["order_id"]]
    with pytest.raises(ValueError, match="Breaking schema drift"):
        registry.check_and_register(removed_col_df, ds_name, "bronze", required_columns=["order_id"])

    # Scenario 3: Incompatible data type change -> Breaking drift
    type_changed_df = added_col_df.copy()
    type_changed_df["amount"] = ["ten", "twenty"]
    with pytest.raises(ValueError, match="Breaking schema drift"):
        registry.check_and_register(type_changed_df, ds_name, "bronze", required_columns=["order_id"])

    # Scenario 4: Missing required column in Data Contract -> Rejected
    missing_req_df = pd.DataFrame({"customer_id": ["c1"]})
    with pytest.raises(ValueError, match="missing required columns"):
        contract_engine.validate_and_split(missing_req_df, "orders", batch_id="test_b1", persist=False)

    # Cleanup test probe records
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM meta.schema_versions WHERE dataset_name = :ds"), {"ds": ds_name})
        conn.execute(text("DELETE FROM meta.incidents WHERE dataset_name = :ds"), {"ds": ds_name})


def test_dq_score_formula_and_quarantine_reconciliation():
    engine = get_engine()
    with engine.connect() as conn:
        dq = conn.execute(
            text(
                """
                SELECT total_records, valid_records, invalid_records, duplicate_records,
                       overall_score, dimension_scores
                FROM meta.quality_results
                WHERE dataset_name = 'order_reviews'
                ORDER BY evaluated_at DESC LIMIT 1
                """
            )
        ).mappings().fetchone()
        quar_cnt = conn.execute(
            text("SELECT COUNT(*) FROM quarantine.quarantined_records WHERE dataset_name = 'order_reviews'")
        ).scalar()

    silver_reviews = DeltaTable("data/silver/order_reviews/_delta").to_pandas()

    # Verify exact conservation of records: valid + invalid == total
    assert dq["valid_records"] + dq["invalid_records"] == dq["total_records"]
    assert len(silver_reviews) == dq["valid_records"] == 98410
    assert int(quar_cnt) == dq["invalid_records"] == 814

    # Independently recalculate arithmetic mean across the 6 DQ dimensions
    dims = dq["dimension_scores"]
    recalculated = round(sum(dims.values()) / len(dims), 2)
    assert recalculated == float(dq["overall_score"]) == 99.73


def test_scd_type_2_half_open_intervals_and_uniqueness():
    engine = get_engine()
    with engine.connect() as conn:
        # 1. Verify exactly ONE current record per customer_unique_id
        multi_current = conn.execute(
            text(
                """
                SELECT customer_unique_id, COUNT(*)
                FROM gold.dimcustomer
                WHERE is_current = TRUE
                GROUP BY customer_unique_id
                HAVING COUNT(*) > 1
                """
            )
        ).fetchall()
        assert len(multi_current) == 0, f"Found business keys with multiple current records: {multi_current}"

        # 2. Verify current records have effective_to IS NULL
        non_null_current = conn.execute(
            text("SELECT COUNT(*) FROM gold.dimcustomer WHERE is_current = TRUE AND effective_to IS NOT NULL")
        ).scalar()
        assert int(non_null_current) == 0

        # 3. Verify historical records have effective_to IS NOT NULL and effective_from <= effective_to
        invalid_hist = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM gold.dimcustomer
                WHERE is_current = FALSE AND (effective_to IS NULL OR effective_from > effective_to)
                """
            )
        ).scalar()
        assert int(invalid_hist) == 0

    # 4. Synthetic address change test proving historical version creation & half-open interval continuity
    cfg = load_yaml_config()
    tracker = ObservabilityTracker(engine)
    lineage = OpenLineageTracker(engine)
    builder = GoldStarSchemaBuilder(engine, cfg, tracker, lineage)

    cust_df = pd.DataFrame(
        [
            {"customer_id": "c1", "customer_unique_id": "u_test", "customer_zip_code_prefix": "01001", "customer_city": "sao paulo", "customer_state": "SP"},
            {"customer_id": "c2", "customer_unique_id": "u_test", "customer_zip_code_prefix": "20040", "customer_city": "rio de janeiro", "customer_state": "RJ"},
        ]
    )
    ord_df = pd.DataFrame(
        [
            {"customer_id": "c1", "order_purchase_timestamp": "2018-01-10 10:00:00"},
            {"customer_id": "c2", "order_purchase_timestamp": "2018-05-20 14:30:00"},
        ]
    )
    scd = builder.build_dim_customer_scd2(cust_df, ord_df)
    assert len(scd) == 2
    v1, v2 = scd.iloc[0], scd.iloc[1]
    assert bool(v1["is_current"]) is False and v1["effective_to"] == v2["effective_from"]
    assert bool(v2["is_current"]) is True and pd.isna(v2["effective_to"])
