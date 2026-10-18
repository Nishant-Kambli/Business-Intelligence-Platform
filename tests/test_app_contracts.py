"""Focused checks for the People Intelligence app's current V1 contracts."""

import io
import unittest
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
from ai_bi.analytics import workforce_kpis
from ai_bi.app_state import dataset_configs
from ai_bi.config import DATA_PATH, MODEL_PATH
from ai_bi.data import load_dataset, normalize_dataset
from ai_bi.dataset_registry import detect_dataset, load_dataset_configs, module_health
from ai_bi.model_registry import active_models, load_model_registry, model_compatibility
from ai_bi.upload_service import process_upload
from ai_bi.model_service import load_model, local_contributions, model_feature_columns, model_input_from_canonical, predict_employee
from ai_bi.report_service import executive_report, excel_bytes, pdf_bytes, report_rows


class DatasetContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = dataset_configs()["workforce_v1"]
        cls.frame = load_dataset(DATA_PATH)
        cls.normalized, cls.report = normalize_dataset(cls.frame, cls.config)

    def test_default_dataset_shape_and_department_repair(self):
        self.assertEqual(self.frame.shape, (1441, 35))
        self.assertNotIn("Nan", self.normalized["department"].astype(str).values)

    def test_schema_detection_and_module_health(self):
        detected, mapping = detect_dataset(self.frame.columns, load_dataset_configs())
        self.assertEqual(detected.dataset_id, "workforce_v1")
        self.assertFalse(mapping["collisions"])
        full_health = module_health(detected, mapping["canonical_to_raw"])
        self.assertTrue(full_health["retention"]["supported"])
        partial, mapping = detect_dataset(["Age", "Department", "JobRole"], load_dataset_configs())
        self.assertEqual(partial.dataset_id, "workforce_v1")
        health = module_health(partial, mapping["canonical_to_raw"])
        self.assertTrue(health["workforce"]["supported"])
        self.assertFalse(health["compensation"]["supported"])
        spec = load_model_registry()["attrition_prediction"]
        partial_source = {raw for aliases in mapping["canonical_to_raw"].values() for raw in aliases}
        compatibility = model_compatibility(spec, mapping["canonical_to_raw"], partial_source)
        self.assertFalse(compatibility["compatible"])

    def test_generic_turnover_aliases_and_modules(self):
        columns = ["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "left", "promotion_last_5years", "Department", "salary"]
        detected, mapping = detect_dataset(columns, load_dataset_configs())
        self.assertEqual(detected.dataset_id, "workforce_v1")
        health = module_health(detected, mapping["canonical_to_raw"])
        self.assertTrue(health["workforce"]["supported"])
        self.assertTrue(health["compensation"]["supported"])
        self.assertTrue(health["engagement"]["supported"])
        self.assertTrue(health["retention"]["supported"])
        spec = load_model_registry()["attrition_prediction"]
        source_columns = {raw for aliases in mapping["canonical_to_raw"].values() for raw in aliases}
        compatibility = model_compatibility(spec, mapping["canonical_to_raw"], source_columns)
        self.assertFalse(compatibility["compatible"])

    def test_training_performance_dataset_upload_is_not_rejected(self):
        sample = pd.DataFrame({
            "Employee ID": [10, 11],
            "BusinessUnit": ["Ops", "Sales"],
            "DepartmentType": ["Production", "Sales"],
            "Performance Score": ["Fully Meets", "Exceeds"],
            "Current Employee Rating": [4, 5],
            "Engagement Score": [3, 2],
            "Satisfaction Score": [4, 3],
            "Training Program Name": ["Safety", "Leadership"],
            "Training Outcome": ["Complete", "Failed"],
            "Training Cost": [120.0, 250.0],
            "Age": [40, 31],
        })
        uploaded = process_upload(sample.to_csv(index=False).encode("utf-8"), load_dataset_configs())
        self.assertIsNone(uploaded.error_code)
        cleaned = normalize_dataset(uploaded.frame, uploaded.config)
        health = cleaned[1]["module_health"]
        self.assertTrue(health["workforce"]["supported"])
        self.assertTrue(health["engagement"]["supported"])
        self.assertTrue(health["performance"]["supported"])
        self.assertTrue(health["training"]["supported"])
        self.assertFalse(health["compensation"]["supported"])
        self.assertFalse(health["retention"]["supported"])
        self.assertFalse(health["recruitment"]["supported"])

    def test_unknown_nonempty_csv_is_still_loaded(self):
        sample = pd.DataFrame({"Office Region!": ["North", "South"], "Cost Center Code": [101, 202]})
        uploaded = process_upload(sample.to_csv(index=False).encode("utf-8"), load_dataset_configs())
        self.assertIsNone(uploaded.error_code)
        normalized, report = normalize_dataset(uploaded.frame, uploaded.config)
        self.assertIn("office_region", normalized)
        self.assertIn("cost_center_code", normalized)
        self.assertTrue(report["module_health"]["executive_dashboard"]["supported"])
        self.assertFalse(report["module_health"]["workforce"]["supported"])

    def test_ordinal_text_is_not_coerced_to_zero(self):
        sample = pd.DataFrame({
            "Job Satisfaction": ["High", "Low"],
            "Work-Life Balance": ["Good", "Poor"],
            "Performance Rating": ["Average", "High"],
            "Job Level": ["Entry", "Senior"],
            "Attrition": ["Stayed", "Left"],
        })
        normalized, report = normalize_dataset(sample, self.config)
        self.assertEqual(set(normalized["job_satisfaction"]), {"High", "Low"})
        self.assertEqual(set(normalized["work_life_balance"]), {"Good", "Poor"})
        self.assertEqual(set(normalized["performance_rating"]), {"Average", "High"})
        self.assertEqual(set(normalized["job_level"]), {"Entry", "Senior"})
        self.assertEqual(set(normalized["attrition"]), {"No", "Yes"})
        self.assertIn("job_satisfaction", report["numeric_fields_preserved_as_categories"])

    def test_workforce_kpis(self):
        metrics = workforce_kpis(self.normalized)
        self.assertEqual(metrics["employees"], 1441)
        self.assertEqual(metrics["departments"], 3)
        self.assertAlmostEqual(metrics["attrition_rate"], 16.308, places=2)


class ExistingAttritionModel(unittest.TestCase):
    def test_saved_pipeline_inference_and_factors(self):
        configs = load_dataset_configs()
        config = configs["workforce_v1"]
        frame, _ = normalize_dataset(load_dataset(DATA_PATH), config)
        model = load_model(MODEL_PATH)
        columns = model_feature_columns(model)
        record = model_input_from_canonical(frame.iloc[0].to_dict(), config, columns)
        result = predict_employee(model, record, columns)
        self.assertIn(result["prediction"], ("Yes", "No"))
        self.assertGreaterEqual(result["probability"], 0)
        self.assertLessEqual(result["probability"], 1)
        self.assertEqual(len(local_contributions(model, record)), 5)
        self.assertEqual(len(columns), 30)

    def test_registry_only_enables_compatible_attrition_model(self):
        registry = load_model_registry()
        self.assertEqual([spec.model_id for spec in active_models()], ["attrition_prediction"])
        spec = registry["attrition_prediction"]
        config = load_dataset_configs()["workforce_v1"]
        frame, report = normalize_dataset(load_dataset(DATA_PATH), config)
        source_columns = {name for aliases in report["canonical_to_raw"].values() for name in aliases}
        result = model_compatibility(spec, report["available_fields"], source_columns)
        self.assertTrue(result["compatible"])
        self.assertEqual(frame.shape[0], 1441)


class ExecutiveReports(unittest.TestCase):
    def test_csv_rows_excel_package_and_pdf(self):
        config = load_dataset_configs()["workforce_v1"]
        frame, _ = normalize_dataset(load_dataset(DATA_PATH), config)
        report = executive_report(frame, config)
        self.assertGreater(len(report_rows(report)), 0)
        workbook = zipfile.ZipFile(io.BytesIO(excel_bytes(report)))
        for name in workbook.namelist():
            if name.endswith(".xml"):
                ET.fromstring(workbook.read(name))
        self.assertTrue(pdf_bytes(report).startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
