

# method1 归一化坐标

    def _regression(self, feat):
        feat = self.regression1(feat)
        feat = feat.view(feat.shape[0], -1)
        feat = self.regression2(feat)
        feat = self.regression(feat)
        return feat

    expected_coords = expected_coords * float(self.window // 2)
    expected_coords = expected_coords *  self.step_fine

## 结果较差

# method2 直接回归坐标


    def _regression(self, feat):
        feat = self.regression1(feat)
        feat = feat.view(feat.shape[0], -1)
        feat = self.regression2(feat)
        feat = self.regression(feat)
        return feat


    expected_coords = self._regression(center_desc)

## 效果较好

# method3 归一化坐标
    def _regression(self, feat):
        feat = self.regression1(feat)
        feat = feat.view(feat.shape[0], -1)
        feat = self.regression2(feat)
        feat = self.regression(feat)
        return feat

    expected_coords = self._regression(center_desc)
        
    W = self.window
    expected_coords = torch.clamp(expected_coords, -W/2, W/2)
    expected_coords = (expected_coords - torch.min(expected_coords)) / (torch.max(expected_coords) - torch.min(expected_coords))
    expected_coords = (expected_coords-0.5).true_divide(0.5)
    expected_coords = expected_coords * float(self.window // 2)
    expected_coords = expected_coords *  self.step_fine
