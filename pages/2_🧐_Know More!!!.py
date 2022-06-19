# Copyright 2018-2022 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to Streamlit! 👋")

    st.sidebar.success("Select a demo above.")


st.warning(""" ###So, what do these categories mean? This is actually explained in the paper linked on the [Know More](https://share.streamlit.io/pranjalpruthi/chest-xray-analysis-ml/main/%F0%9F%93%8A_CheXM_Multi_Class_Prediction.py/Know_More!!!) Section. **👈."):

- AORTIC ENLARGEMENT : 
-- Aortic enlargement is known as a sign of an aortic aneurysm. This condition often occurs in the ascending aorta.
-- In general, the term aneurysm is used when the axial diameter is >5.0 cm for the ascending aorta and >4.0 cm for the descending aorta.


- ATELECTASIS:

-- Atelectasis is a condition where there is no air in part or all of the lungs and they have collapsed.
-- A common cause of atelectasis is obstruction of the bronchi.
-- In atelectasis, there is an increase in density on chest x-ray (usually whiter; black on black-and-white inversion images).

- CALCIFICATION:

--Calcium (calcification) may be deposited in areas where previous inflammation of the lungs or pleura has healed.
--Many diseases or conditions can cause calcification on chest x-ray.
--Calcification may occur in the Aorta (as with atherosclerosis) or it may occur in mediastinal lymph nodes (as with previous infection, tuberculosis, or histoplasmosis).

- CARDIOMEGALY:

--Cardiomegaly is usually diagnosed when the ratio of the heart's width to the width of the chest is more than 50%. This diagnostic criterion may be an essential basis for this competition.
--Cardiomegaly can be caused by many conditions, including hypertension, coronary artery disease, infections, inherited disorders, and cardiomyopathies.
--The heart-to-lung ratio criterion for the diagnosis of cardiomegaly is a ratio of greater than 0.5. However, this is only valid if the XRay is performed while the patient is standing. If the patient is sitting or in bed, this criterion cannot be used. To determine whether a patient is sitting or standing (and consequently whether this criteron is valid), we will detect the presence of air in the stomach (if there is no air in it, the patient is not standing and the criterion cannot be used)

- CONSOLIDATION:

--Consolidation is a decrease in lung permeability due to infiltration of fluid, cells, or tissue replacing the air-containing spaces in the alveoli.
--Consolidation is officially referred to as air space consolidation.
--On X-rays displaying air space consolidation, the lung field's density is increased, and pulmonary blood vessels are not seen, but black bronchi can be seen in the white background, which is called "air bronchogram". Since air remains in the bronchial tubes, they do not absorb X-rays and appear black, and the black and white are reversed from normal lung fields.

- ILD:

--ILD stands for "Interstitial Lung Disease".
--Interstitial Lung Disease is a general term for many conditions in which the interstitial space is injured.
--The interstitial space refers to the walls of the alveoli (air sacs in the lungs) and the space around the blood vessels and small airways.
--Chest radiographic findings include ground-glass opacities (i.e., an area of hazy opacification), linear reticular shadows, and granular shadows.

- INFILTRATION:

--The infiltration of some fluid component into the alveoli causes an infiltrative shadow (Infiltration).
--It is difficult to distinguish from consolidation and, in some cases, impossible to distinguish.


- LUNG OPACITY:

--Lung opacity is a loose term with many potential interpretations/meanings.
--Lung opacity can often be identified as any area in the chest radiograph that is more white than it should be.

- NODULE/MASS:

--Nodules and masses are seen primarily in lung cancer, and metastasis from other parts of the body such as colon cancer and kidney cancer, tuberculosis, pulmonary mycosis, non-tuberculous mycobacterium, obsolete pneumonia, and benign tumors.
--A nodule/mass is a round shade (typically less than 3 cm in diameter – resulting in much smaller than average bounding boxes) that appears on a chest X-ray image.

- Other lesion: Others include all abnormalities that do not fall into any other category. This includes bone penetrating images, fractures, subcutaneous emphysema, etc

- PLEURAL EFFUSION:

--Pleural effusion is the accumulation of water outside the lungs in the chest cavity.
--The outside of the lungs is covered by a thin membrane consisting of two layers known as the pleura. Fluid accumulation between these two layers (chest-wall/parietal-pleura and the lung-tissue/visceral-pleura) is called pleural effusion.
--The findings of pleural effusion vary widely and vary depending on whether the radiograph is taken in the upright or supine position.
--The most common presentation of pleural effusion is elevation of the diaphragm on one side, flattening the diaphragm, or blunting the angle between rib and diaphragm (typically more than 30 degrees)

- PLEURAL THICKENING:

--The pleura is the membrane that covers the lungs, and the change in the thickness of the pleura is called pleural thickening.
--It is often seen in the uppermost part of the lung field (the apex of the lung).

- PNEUMOTHORAX:

--A pneumothorax is a condition in which air leaks from the lungs and accumulates in the chest cavity.
--When air leaks and accumulates in the chest, it cannot expand outward like a balloon due to the ribs' presence. Instead, the lungs are pushed by the air and become smaller. In other words, a pneumothorax is a situation where air leaks from the lungs and the lungs become smaller (collapsed).
--In a chest radiograph of a pneumothorax, the collapsed lung is whiter than normal, and the area where the lung is gone is uniformly black. Besides, the edges of the lung may appear linear.

- PULMONARY FIBROSIS:

--Pulmonary Fibrosis is inflammation of the lung interstitium due to various causes, resulting in thickening and hardening of the walls, fibrosis, and scarring.
--The fibrotic areas lose their air content, which often results in dense cord shadows or granular shadows.

""")


if __name__ == "__main__":
    run()
