import logging

logging.basicConfig(filename=".logs/aih_rag.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')



logger = logging.getLogger(__name__)
