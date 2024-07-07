import unittest
from automl import MLSystem

import warnings
warnings.filterwarnings('ignore')

class TestMLSystem(unittest.TestCase):
    def test_entire_workflow(self):
        # Inicializa el sistema de ML
        system = MLSystem()
        # Ejecuta el flujo de trabajo completo y obtiene el resultado
        result = system.run_entire_workflow()  

        # Verifica que el flujo de trabajo se haya completado con éxito
        self.assertTrue(result['success'], "The ML system workflow should have completed successfully.")

        # Verifica que el AUC del modelo sea razonable
        # Nota: el umbral específico depende del caso de uso y expectativas
        self.assertGreater(result['AUC'], 0.8, "The model AUC should be above 0.8.")

if __name__ == '__main__':
    unittest.main()