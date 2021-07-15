import unittest
import pytest
import os
import sys
import shutil
from typing import Collection, Iterable, List, Union, Dict, Callable
import pythainlp
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)
from thai2transformers.pipelines.inference import (
    TokenClassificationPipeline
)


class TokenClassificationPipelineTest(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None
        self.assertEqual(pythainlp.__version__, '2.2.4')
        self.thainer_ner_pipeline = TokenClassificationPipeline(
                    model=AutoModelForTokenClassification.from_pretrained(
                        'airesearch/wangchanberta-base-att-spm-uncased',
                        revision='finetuned@thainer-ner'
                    ),
                    tokenizer=AutoTokenizer.from_pretrained(
                        'airesearch/wangchanberta-base-att-spm-uncased',
                        revision='finetuned@thainer-ner'
                    )
        )
        self.lst20_ner_pipeline = TokenClassificationPipeline(
                    model=AutoModelForTokenClassification.from_pretrained(
                        'airesearch/wangchanberta-base-att-spm-uncased',
                        revision='finetuned@lst20-ner'
                    ),
                    tokenizer=AutoTokenizer.from_pretrained(
                        'airesearch/wangchanberta-base-att-spm-uncased',
                        revision='finetuned@lst20-ner'
                    ),
                    scheme='IOBE',
                    tag_delimiter='_',
        )
        self.lst20_pos_pipeline = TokenClassificationPipeline(
                    model=AutoModelForTokenClassification.from_pretrained(
                        'airesearch/wangchanberta-base-att-spm-uncased',
                        revision='finetuned@lst20-pos'
                    ),
                    tokenizer=AutoTokenizer.from_pretrained(
                        'airesearch/wangchanberta-base-att-spm-uncased',
                        revision='finetuned@lst20-pos'
                    ),
                    scheme=None,
                    tag_delimiter=None,
        )

    def test_init_pipeline_1(self):

        pipeline = TokenClassificationPipeline(
                    model=AutoModelForTokenClassification.from_pretrained(
                        'airesearch/wangchanberta-base-att-spm-uncased',
                        revision='finetuned@thainer-ner'
                    ),
                    tokenizer=AutoTokenizer.from_pretrained(
                        'airesearch/wangchanberta-base-att-spm-uncased',
                        revision='finetuned@thainer-ner'
                    )
        )

        assert pipeline is not None
        assert pipeline.model is not None
        assert pipeline.tokenizer is not None
        assert pipeline.id2label is not None
        assert pipeline.label2id is not None

    def test_init_pipeline_2(self):

        with self.assertRaises(AssertionError) as context:

            pipeline = TokenClassificationPipeline(
                        model=None,
                        tokenizer=None
            )
            self.assertEqual(type(context.exception), AssertionError)

    def test_init_pipeline_3(self):

        with self.assertRaises(AssertionError) as context:

            pipeline = TokenClassificationPipeline(
                        model=AutoTokenizer.from_pretrained(
                            'airesearch/wangchanberta-base-att-spm-uncased',
                            revision='finetuned@thainer-ner'
                        ),
                        tokenizer=AutoModelForTokenClassification.from_pretrained(
                            'airesearch/wangchanberta-base-att-spm-uncased',
                            revision='finetuned@thainer-ner'
                        )
            )
            self.assertEqual(type(context.exception), AssertionError)


    def test_ner_newmm_pretokenizer(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = False

        sentence = 'เจ้าพ่อสื่อระดับโลก﻿รูเพิร์ท เมอร์ดอค﻿สามารถเข้าซื้อกิจการดาวโจนส์ได้สำเร็จ'
        tokens = ['เจ้า', 'พ่อสื่อ', 'ระดับโลก', '\ufeff', 'รู', 'เพิร์ท', ' ', 'เม', 'อร', '์ดอค\ufeff', 'สามารถ', 'เข้า', 'ซื้อ', 'กิจการ', 'ดาวโจนส์', 'ได้', 'สำเร็จ']
        self.assertEqual(tokens, pipeline.pretokenizer(sentence))

        sentence = '​เกาะสมุยฝนตกน้ำท่วมเตือนห้ามลงเล่นน้ำ'
        tokens = ['\u200b', 'เกาะ', 'สมุย', 'ฝนตก', 'น้ำท่วม', 'เตือน', 'ห้าม', 'ลง', 'เล่น', 'น้ำ']
        self.assertEqual(tokens, pipeline.pretokenizer(sentence))

        sentence = 'http://www.bangkokhealth.com/healthnews_htdoc/healthnews _ detail.asp?Number=10506'
        tokens = ['http', '://', 'www', '.', 'bangkokhealth', '.', 'com', '/', 'healthnews', '_', 'htdoc', '/', 'healthnews', ' ', '_', ' ', 'detail', '.', 'asp', '?', 'Number', '=', '10506']
        self.assertEqual(tokens, pipeline.pretokenizer(sentence))

        sentence = 'สงสัยติดหวัดนก อีกคนยังน่าห่วง'
        tokens = ['สงสัย', 'ติด', 'หวัด', 'นก', ' ', 'อีก', 'คน', 'ยัง', 'น่า', 'ห่วง']
        self.assertEqual(tokens, pipeline.pretokenizer(sentence))


    def test_ner_newmm_preprocess_1(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = False
        
        sentence = 'เจ้าพ่อสื่อระดับโลก﻿รูเพิร์ท เมอร์ดอค﻿สามารถเข้าซื้อกิจการดาวโจนส์ได้สำเร็จ'
        tokens = ['เจ้า', 'พ่อสื่อ', 'ระดับโลก', '\ufeff', 'รู', 'เพิร์ท', space_token, 'เม', 'อร', '์ดอค\ufeff', 'สามารถ', 'เข้า', 'ซื้อ', 'กิจการ', 'ดาวโจนส์', 'ได้', 'สำเร็จ']
        self.assertEqual(tokens, pipeline.preprocess(sentence))

        sentence = '​เกาะสมุยฝนตกน้ำท่วมเตือนห้ามลงเล่นน้ำ'
        tokens = ['\u200b', 'เกาะ', 'สมุย', 'ฝนตก', 'น้ำท่วม', 'เตือน', 'ห้าม', 'ลง', 'เล่น', 'น้ำ']
        self.assertEqual(tokens, pipeline.preprocess(sentence))

        sentence = 'http://www.bangkokhealth.com/healthnews_htdoc/healthnews _ detail.asp?Number=10506'
        tokens = ['http', '://', 'www', '.', 'bangkokhealth', '.', 'com', '/', 'healthnews', '_', 'htdoc', '/', 'healthnews', space_token, '_', space_token, 'detail', '.', 'asp', '?', 'Number', '=', '10506']
        self.assertEqual(tokens, pipeline.preprocess(sentence))

        sentence = 'สงสัยติดหวัดนก   อีกคนยังน่าห่วง'
        tokens = ['สงสัย', 'ติด', 'หวัด', 'นก', space_token, 'อีก', 'คน', 'ยัง', 'น่า', 'ห่วง']
        self.assertEqual(tokens, pipeline.preprocess(sentence))

        sentence = 'ABC สงสัยติดหวัดนก   อีกคนยังน่าห่วง'
        tokens = ['ABC', space_token, 'สงสัย', 'ติด', 'หวัด', 'นก', space_token, 'อีก', 'คน', 'ยัง', 'น่า', 'ห่วง']
        self.assertEqual(tokens, pipeline.preprocess(sentence))

    def test_ner_newmm_preprocess_2(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = False
        
        sentences = ['เจ้าพ่อสื่อระดับโลก﻿รูเพิร์ท เมอร์ดอค﻿สามารถเข้าซื้อกิจการดาวโจนส์ได้สำเร็จ',
                     '​เกาะสมุยฝนตกน้ำท่วมเตือนห้ามลงเล่นน้ำ',
                     'http://www.bangkokhealth.com/healthnews_htdoc/healthnews _ detail.asp?Number=10506',
                     'สงสัยติดหวัดนก   อีกคนยังน่าห่วง',
                     'ABC สงสัยติดหวัดนก   อีกคนยังน่าห่วง']

        list_of_tokens = [['เจ้า', 'พ่อสื่อ', 'ระดับโลก', '\ufeff', 'รู', 'เพิร์ท', space_token, 'เม', 'อร', '์ดอค\ufeff', 'สามารถ', 'เข้า', 'ซื้อ', 'กิจการ', 'ดาวโจนส์', 'ได้', 'สำเร็จ'],
                          ['\u200b', 'เกาะ', 'สมุย', 'ฝนตก', 'น้ำท่วม', 'เตือน', 'ห้าม', 'ลง', 'เล่น', 'น้ำ'],
                          ['http', '://', 'www', '.', 'bangkokhealth', '.', 'com', '/', 'healthnews', '_', 'htdoc', '/', 'healthnews', space_token, '_', space_token, 'detail', '.', 'asp', '?', 'Number', '=', '10506'],
                          ['สงสัย', 'ติด', 'หวัด', 'นก', space_token, 'อีก', 'คน', 'ยัง', 'น่า', 'ห่วง'],
                          ['ABC', space_token, 'สงสัย', 'ติด', 'หวัด', 'นก', space_token, 'อีก', 'คน', 'ยัง', 'น่า', 'ห่วง']]

        self.assertEqual(list_of_tokens, pipeline.preprocess(sentences))

    def test_ner_newmm_preprocess_3(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        
        sentence = 'เจ้าพ่อสื่อระดับโลก﻿รูเพิร์ท เมอร์ดอค﻿สามารถเข้าซื้อกิจการดาวโจนส์ได้สำเร็จ'
        tokens = ['เจ้า', 'พ่อสื่อ', 'ระดับโลก', '\ufeff', 'รู', 'เพิร์ท', space_token, 'เม', 'อร', '์ดอค\ufeff', 'สามารถ', 'เข้า', 'ซื้อ', 'กิจการ', 'ดาวโจนส์', 'ได้', 'สำเร็จ']
        self.assertEqual(tokens, pipeline.preprocess(sentence))

        sentence = '​เกาะสมุยฝนตกน้ำท่วมเตือนห้ามลงเล่นน้ำ'
        tokens = ['\u200b', 'เกาะ', 'สมุย', 'ฝนตก', 'น้ำท่วม', 'เตือน', 'ห้าม', 'ลง', 'เล่น', 'น้ำ']
        self.assertEqual(tokens, pipeline.preprocess(sentence))

        sentence = 'http://www.bangkokhealth.com/healthnews_htdoc/healthnews _ detail.asp?Number=10506'
        tokens = ['http', '://', 'www', '.', 'bangkokhealth', '.', 'com', '/', 'healthnews', '_', 'htdoc', '/', 'healthnews', space_token, '_', space_token, 'detail', '.', 'asp', '?', 'number', '=', '10506']
        self.assertEqual(tokens, pipeline.preprocess(sentence))

        sentence = 'สงสัยติดหวัดนก   อีกคนยังน่าห่วง'
        tokens = ['สงสัย', 'ติด', 'หวัด', 'นก', space_token, 'อีก', 'คน', 'ยัง', 'น่า', 'ห่วง']
        self.assertEqual(tokens, pipeline.preprocess(sentence))

        sentence = 'ABC สงสัยติดหวัดนก   อีกคนยังน่าห่วง'
        tokens = ['abc', space_token, 'สงสัย', 'ติด', 'หวัด', 'นก', space_token, 'อีก', 'คน', 'ยัง', 'น่า', 'ห่วง']
        self.assertEqual(tokens, pipeline.preprocess(sentence))

    def test_ner_newmm_preprocess_4(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        
        sentences = ['เจ้าพ่อสื่อระดับโลก﻿รูเพิร์ท เมอร์ดอค﻿สามารถเข้าซื้อกิจการดาวโจนส์ได้สำเร็จ',
                     '​เกาะสมุยฝนตกน้ำท่วมเตือนห้ามลงเล่นน้ำ',
                     'http://www.bangkokhealth.com/healthnews_htdoc/healthnews _ detail.asp?Number=10506',
                     'สงสัยติดหวัดนก   อีกคนยังน่าห่วง',
                     'ABC สงสัยติดหวัดนก   อีกคนยังน่าห่วง']

        list_of_tokens = [['เจ้า', 'พ่อสื่อ', 'ระดับโลก', '\ufeff', 'รู', 'เพิร์ท', space_token, 'เม', 'อร', '์ดอค\ufeff', 'สามารถ', 'เข้า', 'ซื้อ', 'กิจการ', 'ดาวโจนส์', 'ได้', 'สำเร็จ'],
                          ['\u200b', 'เกาะ', 'สมุย', 'ฝนตก', 'น้ำท่วม', 'เตือน', 'ห้าม', 'ลง', 'เล่น', 'น้ำ'],
                          ['http', '://', 'www', '.', 'bangkokhealth', '.', 'com', '/', 'healthnews', '_', 'htdoc', '/', 'healthnews', space_token, '_', space_token, 'detail', '.', 'asp', '?', 'number', '=', '10506'],
                          ['สงสัย', 'ติด', 'หวัด', 'นก', space_token, 'อีก', 'คน', 'ยัง', 'น่า', 'ห่วง'],
                          ['abc', space_token, 'สงสัย', 'ติด', 'หวัด', 'นก', space_token, 'อีก', 'คน', 'ยัง', 'น่า', 'ห่วง']]

        self.assertEqual(list_of_tokens, pipeline.preprocess(sentences))

    def test_ner_grouped_entities_unstrict_1(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = True
        pipeline.strict = False

        input_ner_tags = [
            ('สถาบัน', 'I-ORGANIZATION'),
            ('วิทย', 'I-ORGANIZATION'),
            ('สิริ', 'I-ORGANIZATION'),
            ('เมธี', 'I-ORGANIZATION'),
            (' ', 'O'),
            ('ตั้งอยู่', 'O'),
            ('ที่', 'O'),
            ('กรุงเทพ', 'I-LOCATION'),
            ('ระยอง', 'B-LOCATION'),
            ('12:00', 'I-TIME'),
            ('น.', 'I-TIME'),
            (' ', 'O'),
            ('จังหวัด', 'I-LOCATION'),
            ('กรุงเทพ', 'I-LOCATION'),
            ('มกราคม', 'B-TIME'),
            ('เดือน', 'B-TIME'),
            ('มกราคม', 'I-TIME'),
            (' ', 'O')
        ]
        
        expected = [
            {'entity_group': 'ORGANIZATION', 'word': 'สถาบันวิทยสิริเมธี'},
            {'entity_group': 'O', 'word': ' ตั้งอยู่ที่'},
            {'entity_group': 'LOCATION', 'word': 'กรุงเทพ'},
            {'entity_group': 'LOCATION', 'word': 'ระยอง'},
            {'entity_group': 'TIME', 'word': '12:00น.'},
            {'entity_group': 'O', 'word': ' '},
            {'entity_group': 'LOCATION', 'word': 'จังหวัดกรุงเทพ'},
            {'entity_group': 'TIME', 'word': 'มกราคม'},
            {'entity_group': 'TIME', 'word': 'เดือนมกราคม'},
            {'entity_group': 'O', 'word': ' '}
        ]
        actual = pipeline._group_entities(input_ner_tags)
        
        self.assertEqual(actual, expected)

    def test_ner_grouped_entities_unstrict_2(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = True
        pipeline.strict = False

        input_ner_tags = [
            ('กรุงเทพ', 'I-LOCATION'),
            ('จังหวัด', 'B-LOCATION'),
            ('กรุงเทพ', 'I-LOCATION'),
            (' ', 'O'),
            ('มกราคม', 'I-TIME'),
        ]
        
        expected = [
            {'entity_group': 'LOCATION', 'word': 'กรุงเทพ'},
            {'entity_group': 'LOCATION', 'word': 'จังหวัดกรุงเทพ'},
            {'entity_group': 'O', 'word': ' '},
            {'entity_group': 'TIME', 'word': 'มกราคม'},
        ]
        actual = pipeline._group_entities(input_ner_tags)
        
        self.assertEqual(actual, expected)

        input_ner_tags = [
            ('กรุงเทพ', 'I-LOCATION'),
            ('จังหวัด', 'B-LOCATION'),
            ('กรุงเทพ', 'I-LOCATION'),
            (' ', 'O'),
            ('มกราคม', 'B-TIME'),
        ]
        
        expected = [
            {'entity_group': 'LOCATION', 'word': 'กรุงเทพ'},
            {'entity_group': 'LOCATION', 'word': 'จังหวัดกรุงเทพ'},
            {'entity_group': 'O', 'word': ' '},
            {'entity_group': 'TIME', 'word': 'มกราคม'},
        ]
        actual = pipeline._group_entities(input_ner_tags)
        self.assertEqual(actual, expected)

        input_ner_tags = [
            ('จังหวัด', 'B-LOCATION'),
            ('กรุงเทพ', 'I-LOCATION'),
            (' ', 'O'),
            ('มกราคม', 'I-TIME'),
        ]
        expected = [
            {'entity_group': 'LOCATION', 'word': 'จังหวัดกรุงเทพ'},
            {'entity_group': 'O', 'word': ' '},
            {'entity_group': 'TIME', 'word': 'มกราคม'},
        ]
        actual = pipeline._group_entities(input_ner_tags)
        self.assertEqual(actual, expected)

        input_ner_tags = [
            ('จังหวัด', 'B-LOCATION'),
            ('กรุงเทพ', 'I-LOCATION'),
            (' ', 'O'),
            ('มกราคม', 'I-TIME'),
            ('ระยอง', 'I-LOCATION'),
        ]
        expected = [
            {'entity_group': 'LOCATION', 'word': 'จังหวัดกรุงเทพ'},
            {'entity_group': 'O', 'word': ' '},
            {'entity_group': 'TIME', 'word': 'มกราคม'},
            {'entity_group': 'LOCATION', 'word': 'ระยอง'},

        ]
        actual = pipeline._group_entities(input_ner_tags)        
        self.assertEqual(actual, expected)

        input_ner_tags = [
            ('จังหวัด', 'B-LOCATION'),
            ('กรุงเทพ', 'I-LOCATION'),
            (' ', 'O'),
            ('มกราคม', 'I-TIME'),
            ('ระยอง', 'I-LOCATION'),
        ]
        expected = [
            {'entity_group': 'LOCATION', 'word': 'จังหวัดกรุงเทพ'},
            {'entity_group': 'O', 'word': ' '},
            {'entity_group': 'TIME', 'word': 'มกราคม'},
            {'entity_group': 'LOCATION', 'word': 'ระยอง'},

        ]
        actual = pipeline._group_entities(input_ner_tags)        
        self.assertEqual(actual, expected)

        input_ner_tags = [
            ('จังหวัด', 'B-LOCATION'),
            ('กรุงเทพ', 'I-LOCATION'),
            (' ', 'O'),
            ('ระยอง', 'I-LOCATION'),
        ]
        expected = [
            {'entity_group': 'LOCATION', 'word': 'จังหวัดกรุงเทพ'},
            {'entity_group': 'O', 'word': ' '},
            {'entity_group': 'LOCATION', 'word': 'ระยอง'},

        ]
        actual = pipeline._group_entities(input_ner_tags)        
        self.assertEqual(actual, expected)

    def test_ner_grouped_entities_unstrict_3(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = True
        pipeline.strict = False
        
        input_ner_tags = [
            ('กรุงเทพ', 'I-LOCATION'),
            ('จังหวัด', 'B-LOCATION'),
            ('กรุงเทพ', 'I-LOCATION'),
            ('มกราคม', 'I-TIME'),
        ]
        expected = [
            {'entity_group': 'LOCATION', 'word': 'กรุงเทพ'},
            {'entity_group': 'LOCATION', 'word': 'จังหวัดกรุงเทพ'},
            {'entity_group': 'TIME', 'word': 'มกราคม'},
        ]
        actual = pipeline._group_entities(input_ner_tags)
        self.assertEqual(actual, expected)

    def test_ner_grouped_entities_unstrict_4(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = True
        pipeline.strict = False
        
        input_ner_tags = [
            ('กรุงเทพ', 'B-LOCATION'),
            (' ', 'O'),
            ('กรุงเทพ', 'I-LOCATION'),
            (' ', 'O'),
        ]
        expected = [
            {'entity_group': 'LOCATION', 'word': 'กรุงเทพ'},
            {'entity_group': 'O', 'word': ' '},
            {'entity_group': 'LOCATION', 'word': 'กรุงเทพ'},
            {'entity_group': 'O', 'word': ' '}
        ]
        actual = pipeline._group_entities(input_ner_tags)
        self.assertEqual(actual, expected)

        input_ner_tags = [
            ('เจนนี่', 'I-PERSON'),
            ('กรุงเทพ', 'I-LOCATION'),
            (' ', 'O'),
        ]
        expected = [
            {'entity_group': 'PERSON', 'word': 'เจนนี่'},
            {'entity_group': 'LOCATION', 'word': 'กรุงเทพ'},
            {'entity_group': 'O', 'word': ' '}
        ]
        actual = pipeline._group_entities(input_ner_tags)
        self.assertEqual(actual, expected)


    def test_ner_ungroup_fix_incorrect_tags_1(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = False
        pipeline.strict = False
        
        input_tags = [ 'I-LOCATION',
            'I-PERSON',
            'O',
            'O',
            'I-PERSON',
            'I-PERSON',
        ]
        expected = [ 'B-LOCATION',
            'B-PERSON',
            'O',
            'O',
            'B-PERSON',
            'I-PERSON',
        ]
        
        actual = pipeline._fix_incorrect_tags(input_tags)
        print('actual:', actual)
        self.assertEqual(actual, expected)


    def test_lst20_ner_grouped_entities_unstrict_1(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = True
        pipeline.strict = False

        input_ner_tags = [
            ('สถาบัน', 'E-ORGANIZATION'),
            ('วิทย', 'I-ORGANIZATION'),
            ('สิริ', 'I-ORGANIZATION'),
            ('เมธี', 'I-ORGANIZATION'),
            (' ', 'O'),
            ('ตั้งอยู่', 'O'),
            ('ที่', 'O'),
            ('กรุงเทพ', 'I-LOCATION'),
            ('ระยอง', 'B-LOCATION'),
            ('12:00', 'I-TIME'),
            ('น.', 'I-TIME'),
            (' ', 'O'),
            ('จังหวัด', 'I-LOCATION'),
            ('กรุงเทพ', 'I-LOCATION'),
            ('มกราคม', 'B-TIME'),
            ('เดือน', 'B-TIME'),
            ('มกราคม', 'I-TIME'),
            (' ', 'O')
        ]
        
        expected = [
            {'entity_group': 'ORGANIZATION', 'word': 'สถาบันวิทยสิริเมธี'},
            {'entity_group': 'O', 'word': ' ตั้งอยู่ที่'},
            {'entity_group': 'LOCATION', 'word': 'กรุงเทพ'},
            {'entity_group': 'LOCATION', 'word': 'ระยอง'},
            {'entity_group': 'TIME', 'word': '12:00น.'},
            {'entity_group': 'O', 'word': ' '},
            {'entity_group': 'LOCATION', 'word': 'จังหวัดกรุงเทพ'},
            {'entity_group': 'TIME', 'word': 'มกราคม'},
            {'entity_group': 'TIME', 'word': 'เดือนมกราคม'},
            {'entity_group': 'O', 'word': ' '}
        ]
        actual = pipeline._group_entities(input_ner_tags)
        
        self.assertEqual(actual, expected)

    
    def test_lst20_ner_grouped_entities_unstrict_2(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = True
        pipeline.strict = False

        input_ner_tags = [
            ('กรุงเทพ', 'E-LOCATION'),
            ('จังหวัด', 'B-LOCATION'),
            ('กรุงเทพ', 'I-LOCATION'),
            (' ', 'O'),
            ('มกราคม', 'E-TIME'),
        ]
        
        expected = [
            {'entity_group': 'LOCATION', 'word': 'กรุงเทพ'},
            {'entity_group': 'LOCATION', 'word': 'จังหวัดกรุงเทพ'},
            {'entity_group': 'O', 'word': ' '},
            {'entity_group': 'TIME', 'word': 'มกราคม'},
        ]
        actual = pipeline._group_entities(input_ner_tags)
        
        self.assertEqual(actual, expected)

        input_ner_tags = [
            ('กรุงเทพ', 'I-LOCATION'),
            ('จังหวัด', 'B-LOCATION'),
            ('กรุงเทพ', 'I-LOCATION'),
            ('กรุงเทพ', 'B-LOCATION'),
            ('มกราคม', 'E-TIME'),
        ]
        
        expected = [
            {'entity_group': 'LOCATION', 'word': 'กรุงเทพ'},
            {'entity_group': 'LOCATION', 'word': 'จังหวัดกรุงเทพ'},
            {'entity_group': 'LOCATION', 'word': 'กรุงเทพ'},
            {'entity_group': 'TIME', 'word': 'มกราคม'},
        ]
        actual = pipeline._group_entities(input_ner_tags)
        self.assertEqual(actual, expected)

        input_ner_tags = [
            ('จังหวัด', 'B-LOCATION'),
            ('กรุงเทพ', 'I-LOCATION'),
            (' ', 'O'),
            ('มกราคม', 'I-TIME'),
        ]
        expected = [
            {'entity_group': 'LOCATION', 'word': 'จังหวัดกรุงเทพ'},
            {'entity_group': 'O', 'word': ' '},
            {'entity_group': 'TIME', 'word': 'มกราคม'},
        ]
        actual = pipeline._group_entities(input_ner_tags)
        self.assertEqual(actual, expected)

        input_ner_tags = [
            ('จังหวัด', 'B-LOCATION'),
            ('กรุงเทพ', 'I-LOCATION'),
            (' ', 'O'),
            ('มกราคม', 'I-TIME'),
            ('ระยอง', 'I-LOCATION'),
        ]
        expected = [
            {'entity_group': 'LOCATION', 'word': 'จังหวัดกรุงเทพ'},
            {'entity_group': 'O', 'word': ' '},
            {'entity_group': 'TIME', 'word': 'มกราคม'},
            {'entity_group': 'LOCATION', 'word': 'ระยอง'},

        ]
        actual = pipeline._group_entities(input_ner_tags)        
        self.assertEqual(actual, expected)

        input_ner_tags = [
            ('จังหวัด', 'B-LOCATION'),
            ('กรุงเทพ', 'I-LOCATION'),
            (' ', 'O'),
            ('มกราคม', 'I-TIME'),
            ('ระยอง', 'I-LOCATION'),
        ]
        expected = [
            {'entity_group': 'LOCATION', 'word': 'จังหวัดกรุงเทพ'},
            {'entity_group': 'O', 'word': ' '},
            {'entity_group': 'TIME', 'word': 'มกราคม'},
            {'entity_group': 'LOCATION', 'word': 'ระยอง'},

        ]
        actual = pipeline._group_entities(input_ner_tags)        
        self.assertEqual(actual, expected)

        input_ner_tags = [
            ('จังหวัด', 'B-LOCATION'),
            ('กรุงเทพ', 'I-LOCATION'),
            (' ', 'O'),
            ('ระยอง', 'I-LOCATION'),
        ]
        expected = [
            {'entity_group': 'LOCATION', 'word': 'จังหวัดกรุงเทพ'},
            {'entity_group': 'O', 'word': ' '},
            {'entity_group': 'LOCATION', 'word': 'ระยอง'},

        ]
        actual = pipeline._group_entities(input_ner_tags)        
        self.assertEqual(actual, expected)

    def test_lst20_ner_grouped_entities_unstrict_4(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = True
        pipeline.strict = False
        
        input_ner_tags = [
            ('กรุงเทพ', 'B-LOCATION'),
            (' ', 'O'),
            ('กรุงเทพ', 'I-LOCATION'),
            (' ', 'O'),
        ]
        expected = [
            {'entity_group': 'LOCATION', 'word': 'กรุงเทพ'},
            {'entity_group': 'O', 'word': ' '},
            {'entity_group': 'LOCATION', 'word': 'กรุงเทพ'},
            {'entity_group': 'O', 'word': ' '}
        ]
        actual = pipeline._group_entities(input_ner_tags)
        self.assertEqual(actual, expected)

        input_ner_tags = [
            ('เจนนี่', 'I-PERSON'),
            ('กรุงเทพ', 'I-LOCATION'),
            (' ', 'O'),
        ]
        expected = [
            {'entity_group': 'PERSON', 'word': 'เจนนี่'},
            {'entity_group': 'LOCATION', 'word': 'กรุงเทพ'},
            {'entity_group': 'O', 'word': ' '}
        ]
        actual = pipeline._group_entities(input_ner_tags)
        self.assertEqual(actual, expected)

    def test_ner_grouped_entities_strict_1(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = True
        pipeline.strict = True

        input_ner_tags = [('สถาบัน', 'B-ORGANIZATION'),
            ('วิทย', 'I-ORGANIZATION'),
            ('สิริ', 'I-ORGANIZATION'),
            ('เมธี', 'I-ORGANIZATION'),
            (' ', 'O'),
            ('ตั้งอยู่', 'O'),
            ('ที่', 'O'),
            ('กรุงเทพ', 'B-LOCATION'),
            ('ระยอง', 'B-LOCATION'),
            ('12:00', 'B-TIME'),
            ('น.', 'I-TIME'),
            (' ', 'O'),
            ('จังหวัด', 'B-LOCATION'),
            ('กรุงเทพ', 'I-LOCATION'),
            ('มกราคม', 'B-TIME'),
            ('เดือน', 'B-TIME'),
            ('มกราคม', 'I-TIME'),
            ]
        expected = [
            {'entity_group': 'ORGANIZATION', 'word': 'สถาบันวิทยสิริเมธี'},
            {'entity_group': 'O', 'word': ' ตั้งอยู่ที่'},
            {'entity_group': 'LOCATION', 'word': 'กรุงเทพ'},
            {'entity_group': 'LOCATION', 'word': 'ระยอง'},
            {'entity_group': 'TIME', 'word': '12:00น.'},
            {'entity_group': 'O', 'word': ' '},
            {'entity_group': 'LOCATION', 'word': 'จังหวัดกรุงเทพ'},
            {'entity_group': 'TIME', 'word': 'มกราคม'},
            {'entity_group': 'TIME', 'word': 'เดือนมกราคม'}
        ]
        actual = pipeline._group_entities(input_ner_tags)
        
        self.assertEqual(actual, expected)

    def test_ner_grouped_entities_strict_2(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = True
        pipeline.strict = True

        input_ner_tags = [('สถาบัน', 'I-ORGANIZATION'),
            ('วิทย', 'I-ORGANIZATION'),
            ('สิริ', 'I-ORGANIZATION'),
            ('เมธี', 'I-ORGANIZATION'),
            (' ', 'O'),
            ('ตั้งอยู่', 'O'),
            ('ที่', 'O'),
            ('กรุงเทพ', 'B-LOCATION'),
            ('ระยอง', 'B-LOCATION'),
            ('12:00', 'B-TIME'),
            ('น.', 'I-TIME'),
            (' ', 'O'),
            ('จังหวัด', 'B-LOCATION'),
            ('กรุงเทพ', 'I-LOCATION'),
            ('มกราคม', 'B-TIME'),
            ('เดือน', 'B-TIME'),
            ('มกราคม', 'I-TIME'),
            ]
        expected = [
            {'entity_group': 'O', 'word': 'สถาบันวิทยสิริเมธี ตั้งอยู่ที่'},
            {'entity_group': 'LOCATION', 'word': 'กรุงเทพ'},
            {'entity_group': 'LOCATION', 'word': 'ระยอง'},
            {'entity_group': 'TIME', 'word': '12:00น.'},
            {'entity_group': 'O', 'word': ' '},
            {'entity_group': 'LOCATION', 'word': 'จังหวัดกรุงเทพ'},
            {'entity_group': 'TIME', 'word': 'มกราคม'},
            {'entity_group': 'TIME', 'word': 'เดือนมกราคม'}
        ]
        actual = pipeline._group_entities(input_ner_tags)
        
        self.assertEqual(actual, expected)

    def test_ner_grouped_entities_strict_3(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = True
        pipeline.strict = True

        input_ner_tags = [
            ('ปี', 'B-TIME'),
            (' ', 'I-TIME'),
            ('พ.ศ.', 'I-TIME'),
            (' ', 'I-TIME'),
            ('2211', 'I-TIME')
            ]
        expected = [
            {'entity_group': 'TIME', 'word': 'ปี พ.ศ. 2211'}
        ]
        actual = pipeline._group_entities(input_ner_tags)
        self.assertEqual(actual, expected)

    def test_ner_merge_pred(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = False
        pipeline.strict = False

        preds = [
            ('บ็อบ', 'B-PERSON'),
            ('เดิน', 'O'),
            ('ทาง', 'O'),
            ('ใน', 'O'),
            ('เดือ', 'B-DATE'),
            ('น', 'B-DATE'),
            ('มกรา', 'I-DATE'),
            ('คม', 'I-DATE')
        ]
        ids = [[0], [1,2], [3], [4,5], [6,7] ]
        expected = [  ('บ็อบ', 'B-PERSON'),
            ('เดินทาง', 'O'),
            ('ใน', 'O'),
            ('เดือน', 'B-DATE'),
            ('มกราคม', 'I-DATE')]
        actual = pipeline._merged_pred(preds, ids)
        self.assertEqual(actual, expected)

    def test_lst20_ner_newmm_inference_ungrouped_entities_1(self):
        space_token =  '<_>'
        pipeline = self.lst20_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = False
        pipeline.strict = False
        pipeline.tag_delimiter = '_'

        sentence = '​เกาะสมุยฝนตกน้ำท่วมเตือนห้ามลงเล่นน้ำ'
        expected = [{'word': 'เกาะ', 'entity':  'B_LOC'},
                    {'word': 'สมุย', 'entity':  'E_LOC'},
                    {'word': 'ฝนตก', 'entity':  'O'},
                    {'word': 'น้ําท่วม', 'entity':  'O'},
                    {'word': 'เตือน', 'entity':  'O'},
                    {'word': 'ห้าม', 'entity':  'O'},
                    {'word': 'ลง', 'entity':  'O'},
                    {'word': 'เล่น', 'entity':  'O'},
                    {'word': 'น้ํา', 'entity':  'O'}]

        actual = pipeline(sentence)
        self.assertEqual(actual, expected)

        sentence = 'สถาบันวิทยสิริเมธี ตั้งอยู่ในจังหวัดระยอง ก่อตั้งขึ้นเมื่อปี พ.ศ. 2558'
        expected = [{'word': 'สถาบันวิทยสิริเมธี', 'entity':  'B_ORG'},
                    {'word': ' ', 'entity': 'O'},
                    {'word': 'ตั้งอยู่', 'entity': 'O'},
                    {'word': 'ใน', 'entity':  'O'},
                    {'word': 'จังหวัด', 'entity':  'B_LOC'},
                    {'word': 'ระยอง', 'entity':  'E_LOC'},
                    {'word': ' ', 'entity':  'O'},
                    {'word': 'ก่อ', 'entity':  'O'},
                    {'word': 'ตั้งขึ้น', 'entity':  'O'},
                    {'word': 'เมื่อ', 'entity':  'O'},
                    {'word': 'ปี', 'entity':  'B_DTM'},
                    {'word': ' ', 'entity':  'I_DTM'},
                    {'word': 'พ.ศ.', 'entity':  'I_DTM'},
                    {'word': ' ', 'entity':  'I_DTM'},
                    {'word': '2558', 'entity':  'E_DTM'}]
        
        expected_tokens = list(map(lambda x: x['word'], expected))
        self.assertEqual(pipeline.pretokenizer(sentence), expected_tokens)
        actual = pipeline(sentence)
        self.assertEqual(actual, expected)

    def test_lst20_ner_newmm_inference_grouped_entities_1(self):

        space_token =  '<_>'
        pipeline = self.lst20_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = True
        pipeline.strict = True
        pipeline.tag_delimiter = '_'

        sentence = '​เกาะสมุย ฝนตก แล้ว'
        expected = [{'word': 'เกาะสมุย', 'entity_group':  'LOC'},
                    {'word': ' ฝนตก แล้ว', 'entity_group':  'O'}]


        actual = pipeline(sentence)
        self.assertEqual(actual, expected)

        sentence = 'สถาบันวิทยสิริเมธี ตั้งอยู่ในจังหวัดระยอง ก่อตั้งขึ้นเมื่อปี พ.ศ. 2558'
        expected = [{'word': 'สถาบันวิทยสิริเมธี', 'entity_group':  'ORG'},
                    {'word': ' ตั้งอยู่ใน', 'entity_group': 'O'},
                    {'word': 'จังหวัดระยอง', 'entity_group':  'LOC'},
                    {'word': ' ก่อตั้งขึ้นเมื่อ', 'entity_group':  'O'},
                    {'word': 'ปี พ.ศ. 2558', 'entity_group':  'DTM'}]
        
        actual = pipeline(sentence)
        self.assertEqual(actual, expected)

    def test_lst20_pos_newmm_inference_1(self):

        space_token =  '<_>'
        pipeline = self.lst20_pos_pipeline
        pipeline.scheme = None
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = False
        pipeline.strict = True
        pipeline.tag_delimiter = ''

        sentence = 'จีน-อินเดียเสี่ยงสูญเสียจากภัยธรรมชาติมากสุด'
        expected_tokens = ['จีน', '-', 'อินเดีย', 'เสี่ยง', 'สูญเสีย',
                           'จาก', 'ภัยธรรมชาติ', 'มาก', 'สุด']

        self.assertEqual(pipeline.pretokenizer(sentence), expected_tokens)

        expected_result = [{'word': 'จีน',	'entity': 'NN'},
                            {'word': '-',	'entity': 'PU'},
                            {'word': 'อินเดีย',	'entity': 'NN'},
                            {'word': 'เสี่ยง',	'entity': 'VV'},
                            {'word': 'สูญเสีย',	'entity': 'VV'},
                            {'word': 'จาก',	'entity': 'PS'},
                            {'word': 'ภัยธรรมชาติ',	'entity': 'NN'},
                            {'word': 'มาก',	'entity': 'VV'},
                            {'word': 'สุด',	'entity': 'AV'}
                           ]
        actual = pipeline(sentence)

        self.assertEqual(actual,expected_result)

    def test_lst20_pos_newmm_inference_2(self):

        space_token =  '<_>'
        pipeline = self.lst20_pos_pipeline
        pipeline.scheme = None
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = True
        pipeline.strict = True
        pipeline.tag_delimiter = ''

        sentence = 'จีน-อินเดียเสี่ยงสูญเสียจากภัยธรรมชาติมากสุด'
        expected_tokens = ['จีน', '-', 'อินเดีย', 'เสี่ยง', 'สูญเสีย',
                           'จาก', 'ภัยธรรมชาติ', 'มาก', 'สุด']

        self.assertEqual(pipeline.pretokenizer(sentence), expected_tokens)

        expected_result = [{'word': 'จีน',	'entity': 'NN'},
                            {'word': '-',	'entity': 'PU'},
                            {'word': 'อินเดีย',	'entity': 'NN'},
                            {'word': 'เสี่ยง',	'entity': 'VV'},
                            {'word': 'สูญเสีย',	'entity': 'VV'},
                            {'word': 'จาก',	'entity': 'PS'},
                            {'word': 'ภัยธรรมชาติ',	'entity': 'NN'},
                            {'word': 'มาก',	'entity': 'VV'},
                            {'word': 'สุด',	'entity': 'AV'}
                           ]
        actual = pipeline(sentence)

        self.assertEqual(actual,expected_result)


    def test_ner_newmm_inference_ungrouped_entities_1(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = False
        pipeline.strict = True

        sentence = '​เกาะสมุยฝนตกน้ำท่วมเตือนห้ามลงเล่นน้ำ'
        expected = [{'word': 'เกาะ', 'entity':  'B-LOCATION'},
                    {'word': 'สมุย', 'entity':  'I-LOCATION'},
                    {'word': 'ฝนตก', 'entity':  'O'},
                    {'word': 'น้ําท่วม', 'entity':  'O'},
                    {'word': 'เตือน', 'entity':  'O'},
                    {'word': 'ห้าม', 'entity':  'O'},
                    {'word': 'ลง', 'entity':  'O'},
                    {'word': 'เล่น', 'entity':  'O'},
                    {'word': 'น้ํา', 'entity':  'O'}]
        actual = pipeline(sentence)
        self.assertEqual(actual, expected)

        sentence = 'สถาบันวิทยสิริเมธี ตั้งอยู่ในพื้นที่วังจันทร์วัลเลย์ ต.ป่ายุบใน จังหวัดระยอง ก่อตั้งขึ้นเมื่อปี พ.ศ. 2558'
        expected = [{'word': 'สถาบันวิทยสิริเมธี', 'entity':  'B-ORGANIZATION'},
                    {'word': ' ', 'entity': 'O'},
                    {'word': 'ตั้งอยู่', 'entity': 'O'},
                    {'word': 'ใน', 'entity':  'O'},
                    {'word': 'พื้นที่', 'entity':  'O'},
                    {'word': 'วัง', 'entity':  'B-LOCATION'},
                    {'word': 'จันทร์', 'entity':  'I-LOCATION'},
                    {'word': 'วัล', 'entity':  'I-LOCATION'},
                    {'word': 'เลย', 'entity':  'I-LOCATION'},
                    {'word': '์', 'entity':  'I-LOCATION'},
                    {'word': ' ', 'entity':  'O'},
                    {'word': 'ต.', 'entity': 'B-LOCATION'},
                    {'word': 'ป่า', 'entity':  'I-LOCATION'},
                    {'word': 'ยุบ', 'entity':  'I-LOCATION'},
                    {'word': 'ใน', 'entity':  'I-LOCATION'},
                    {'word': ' ', 'entity':  'O'},
                    {'word': 'จังหวัด', 'entity':  'B-LOCATION'},
                    {'word': 'ระยอง', 'entity':  'I-LOCATION'},
                    {'word': ' ', 'entity':  'O'},
                    {'word': 'ก่อ', 'entity':  'O'},
                    {'word': 'ตั้งขึ้น', 'entity':  'O'},
                    {'word': 'เมื่อ', 'entity':  'O'},
                    {'word': 'ปี', 'entity':  'O'},
                    {'word': ' ', 'entity':  'O'},
                    {'word': 'พ.ศ.', 'entity':  'B-DATE'},
                    {'word': ' ', 'entity':  'I-DATE'},
                    {'word': '2558', 'entity':  'I-DATE'}]
        
        expected_tokens = list(map(lambda x: x['word'], expected))
        self.assertEqual(pipeline.pretokenizer(sentence), expected_tokens)
        actual = pipeline(sentence)
        self.assertEqual(actual, expected)


    def test_ner_newmm_inference_grouped_entities(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = True
        pipeline.strict = True

        sentence = 'เกาะสมุยฝนตกน้ําท่วมเตือนห้ามลงเล่นน้ํา'
        assert sentence[len('เกาะสมุย'):] == 'ฝนตกน้ําท่วมเตือนห้ามลงเล่นน้ํา'
        expected = [{'word': 'เกาะสมุย', 'entity_group': 'LOCATION'},
                    {'word': 'ฝนตกน้ําท่วมเตือนห้ามลงเล่นน้ํา', 'entity_group': 'O'}]
        actual = pipeline(sentence)

        self.assertEqual(actual, expected)

        sentence = 'เกาะสมุยฝนตกน้ําท่วมเตือนห้าม ลงเล่นน้ํา'
        assert sentence[len('เกาะสมุย'):] == 'ฝนตกน้ําท่วมเตือนห้าม ลงเล่นน้ํา'
        expected = [{'word': 'เกาะสมุย', 'entity_group': 'LOCATION'},
                    {'word': 'ฝนตกน้ําท่วมเตือนห้าม ลงเล่นน้ํา', 'entity_group': 'O'}]
        actual = pipeline(sentence)

        self.assertEqual(actual, expected)
        

    def test_ner_newmm_inference_grouped_entities_multiple_sentences(self):
        space_token =  '<_>'
        pipeline = self.thainer_ner_pipeline
        pipeline.space_token = space_token
        pipeline.lowercase = True
        pipeline.group_entities = True

        sentences = ['เกาะสมุยฝนตกน้ําท่วมเตือนห้ามลงเล่นน้ํา', 'ในเกาะสมุยมีฝนตก']
        
        expected = [[{'word': 'เกาะสมุย', 'entity_group': 'LOCATION'},
                     {'word': 'ฝนตกน้ําท่วมเตือนห้ามลงเล่นน้ํา', 'entity_group': 'O'}],
                    [{'word': 'ใน', 'entity_group': 'O'},
                     {'word': 'เกาะสมุย', 'entity_group': 'LOCATION'},
                     {'word': 'มีฝนตก', 'entity_group': 'O'}]]
        actual = pipeline(sentences)

        self.assertEqual(actual, expected)