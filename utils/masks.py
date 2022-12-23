
'''
Copyright 2022 Andrea Rafanelli.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License
'''

__author__ = 'Andrea Rafanelli'

class getData:
    
    def __init__(self, x_paths,):

        self.x_paths = x_paths
        self.y_paths = [x.replace("img", "label").replace(".jpg", "_lab.png") for x in self.x_paths]

    def __len__(self):

        return len(self.x_paths)


    def __getitem__(self, index):

        image = Image.open(self.x_paths[index])
        mask = Image.open(self.y_paths[index])

        return image, mask, index
