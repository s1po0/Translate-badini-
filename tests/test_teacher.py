import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import teacher


class TeacherTests(unittest.TestCase):
    def test_example_with_student(self):
        student = SimpleNamespace(own_languages={'en'}, translate=lambda text, lang, target: 'attempt')
        with patch.object(teacher, 'ask_teacher', side_effect=[
            {'source_text': 'hello', 'badini': 'initial'}, {'badini': 'سلاڤ'}
        ]) as ask:
            result = teacher.example('local', 'en', student)
        self.assertEqual(result['badini'], 'سلاڤ')
        self.assertEqual(result['student_translation'], 'attempt')
        self.assertEqual(result['approved'], '')
        self.assertEqual(ask.call_count, 2)

    def test_student_without_language(self):
        student = SimpleNamespace(own_languages={'en'})
        with patch.object(teacher, 'ask_teacher', return_value={'source_text': 'سڵاو', 'badini': 'سلاڤ'}) as ask:
            result = teacher.example('local', 'sorani', student)
        self.assertEqual(result['student_translation'], '')
        self.assertEqual(ask.call_count, 1)

    def test_generation_and_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / 'review.csv', Path(tmp) / 'approved.csv'
            args = SimpleNamespace(student_model=None, student_offline=False, output=str(source), model='local', count=2, languages=['en', 'ar'])
            with patch.object(teacher, 'ask_teacher', return_value={'source_text': 'hello', 'badini': 'سلاڤ'}):
                teacher.generate(args)
            with source.open(encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([row['approved'] for row in rows], ['', ''])
            with self.assertRaises(ValueError):
                teacher.export_approved(source, out)
            rows[1]['approved'] = 'yes'
            with source.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=teacher.FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            teacher.export_approved(source, out)
            with out.open(encoding='utf-8', newline='') as f:
                approved = list(csv.DictReader(f))
            self.assertEqual(len(approved), 1)
            self.assertEqual(approved[0]['source_lang'], 'ar')
            with self.assertRaises(ValueError):
                teacher.export_approved(source, out)

    def test_teacher_malformed(self):
        with patch.object(teacher, 'ask_teacher', return_value={'source_text': 'hello'}):
            with self.assertRaises(ValueError):
                teacher.example('local', 'en')
