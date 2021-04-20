#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   ms.py
@Author :   馒头
@Time   :   2020/1/20 9:41
@Contact:   songjian@westlake.edu.cn
@intro  :   class about reading, processing the mzML
'''
import time
import numpy as np
import multiprocessing as mp
from decimal import Decimal
import pyteomics.mzxml
import pyteomics.mzml
import pyteomics.mass
from pyprophet.optimized import sj_find_ok_matches

try:
    profile
except NameError:
    profile = lambda x: x

class mz_Reader():

    def __init__(self, fpath, acquisition_type):
        self.mz = None
        self.all_rt = np.array([])
        self.TimeUnit = None
        self.SwathSettings = np.array([])

        # type
        fpath = str(fpath)
        self.suffix = fpath.split('.')[-1].lower()
        self.acquisition_type = acquisition_type

        self.load(fpath)
        self.init()

    def size(self):
        return len(self.mz)

    def load(self, file_path):
        if self.suffix == 'mzxml':
            self.mz = pyteomics.mzxml.MzXML(file_path, use_index=True)
        elif self.suffix == 'mzml':
            self.mz = pyteomics.mzml.MzML(file_path, use_index=True)

    def init(self):
        self.get_time_unit()
        self.__load_to_memory()
        if self.acquisition_type == 'DIA':
            self.__init_swath_window_array()
            self.check()

    def check(self):
        'MS1+N-MS2'
        assert len(self.all_levels) % len(self.SwathSettings) == 0, 'Swath scan nums != K*(MS1+N-MS2)'

    def get_time_unit(self):
        if self.suffix == 'mzxml':
            if self.mz[self.size() - 1]['retentionTime'] < 3 * 60:
                self.TimeUnit = 'minute'
            else:
                self.TimeUnit = 'second'
        elif self.suffix == 'mzml':
            if self.mz[self.size() - 1]['scanList']['scan'][0]['scan start time'] < 5 * 60:
                self.TimeUnit = 'minute'
            else:
                self.TimeUnit = 'second'

    def process_worker(self, idx_start, idx_end): # 作为进程worker不能私有
        rts, levels, peaks_mz, peaks_intensity = [], [], [], []

        for idx in range(idx_start, idx_end):
            scan = self.mz[idx]
            if self.suffix == 'mzxml':
                rts.append(np.float32(scan['retentionTime']))
                levels.append(np.int8(scan['msLevel']))
            elif self.suffix == 'mzml':
                rts.append(np.float32(scan['scanList']['scan'][0]['scan start time']))
                levels.append(np.int8(scan['ms level']))
            peaks_mz.append(scan['m/z array'].astype(np.float32))
            peaks_intensity.append(scan['intensity array'].astype(np.float32))

        return {'rt': rts, 'level': levels, 'mz': peaks_mz, 'intensity': peaks_intensity}

    def __load_to_memory(self):
        self.all_mz, self.all_intensity, self.all_levels, self.all_rt = [], [], [], []

        # multiprocess load data
        cpu_num = mp.cpu_count()
        process_num = int(cpu_num / 2)  # default cores / 2
        process_num = 8 if process_num >= 8 else process_num
        pool = mp.Pool(process_num)
        slices = np.ceil(np.linspace(0, len(self.mz), process_num+1)).astype(int)

        results = [pool.apply_async(self.process_worker, args=(slices[i], slices[i+1])) for i in range(process_num)]
        results = [r.get() for r in results] # result is dict
        pool.close()
        pool.join()

        for result in results:
            self.all_rt.extend(result['rt'])
            self.all_levels.extend(result['level'])
            self.all_mz.extend(result['mz'])
            self.all_intensity.extend(result['intensity'])

        self.all_rt = np.array(self.all_rt)
        self.all_levels = np.array(self.all_levels)
        self.all_mz = np.array(self.all_mz)
        self.all_intensity = np.array(self.all_intensity)

        if self.acquisition_type == 'DIA':
            # get MS2 window number
            self.raw_ms1_idx = np.where(self.all_levels == 1)[0]
            cycles_ms2_num = np.diff(self.raw_ms1_idx) - 1
            self.windows_num = np.bincount(cycles_ms2_num).argmax()

            # remove incomplete cycle
            bad_slice = []

            # boundary 1：start of cycle
            if self.raw_ms1_idx[0] != 0:
                bad_slice.extend(range(0, self.raw_ms1_idx[0]))
            # boundary 2：end of cycle
            if len(self.mz) - self.raw_ms1_idx[-1] != self.windows_num + 1:
                bad_slice.extend(range(self.raw_ms1_idx[-1], len(self.mz)))
            # others cycle
            for cycle_idx in np.where(cycles_ms2_num != self.windows_num)[0]:
                bad_slice.extend(range(self.raw_ms1_idx[cycle_idx], self.raw_ms1_idx[cycle_idx+1]))

            if len(bad_slice) > 0:
                good_slice = np.arange(len(self.all_levels))
                good_slice = good_slice[~np.isin(good_slice, bad_slice)]

                self.all_rt = self.all_rt[good_slice]
                self.all_levels = self.all_levels[good_slice]
                self.all_mz = self.all_mz[good_slice]
                self.all_intensity = self.all_intensity[good_slice]

            # assign non-sense values to empty
            all_scans_len = np.array(list(map(len, self.all_mz)), dtype=np.int32)
            for zero_idx in np.where(all_scans_len == 0)[0]:
                self.all_mz[zero_idx] = np.array([888.], dtype=np.float32)
                self.all_intensity[zero_idx] = np.array([0.], dtype=np.float32)

        self.all_ms1_idx = np.where(self.all_levels == 1)[0]

        # time
        if self.TimeUnit == 'minute':
            self.all_rt = self.all_rt * 60.

    def get_ms2_mz_range(self):
        ms2_mz = self.all_mz[self.all_levels == 2]
        ms2_mz_max = max(map(max, ms2_mz))
        ms2_mz_min = min(map(min, ms2_mz))
        return (ms2_mz_min, ms2_mz_max)

    def get_scan_level(self, idx):
        return self.all_levels[idx]

    def get_scan_rt(self, idx):
        """时间统一到second单位"""
        return self.all_rt[idx]

    def get_ms1_all_rt(self):
        num_windows = len(self.SwathSettings) - 1
        scans_ms1_rt = self.all_rt[::(num_windows + 1)]
        return scans_ms1_rt

    def get_scan_mz(self, idx):
        return self.all_mz[idx]

    def get_scan_intensity(self, idx):
        return self.all_intensity[idx]

    def get_scan_peaks(self, idx):
        '''Peaks: [mz, intensity]'''
        mz = self.get_scan_mz(idx)
        inten = self.get_scan_intensity(idx)
        return (mz, inten)

    def get_scan_idx_by_rt(self, rt):
        return int((np.abs(self.all_rt - rt).argmin()))

    def get_current_scan_window(self, idx):
        idx = int(idx)

        if self.suffix == 'mzxml':
            middle = Decimal(str(self.mz[idx]['precursorMz'][0]['precursorMz']))
            width = Decimal(str(self.mz[idx]['precursorMz'][0]['windowWideness']))
            return (float(middle - width / 2), float(middle + width / 2))

        elif self.suffix == 'mzml':
            middle = Decimal(
                str(self.mz[idx]['precursorList']['precursor'][0]['isolationWindow']['isolation window target m/z']))
            lower_offset = Decimal(str(
                self.mz[idx]['precursorList']['precursor'][0]['isolationWindow']['isolation window lower offset']))
            upper_offset = Decimal(str(
                self.mz[idx]['precursorList']['precursor'][0]['isolationWindow']['isolation window upper offset']))
            return (float(middle - lower_offset), float(middle + upper_offset))

    def __init_swath_window_array(self):
        """
        返回一个array，表示的是isolation window的起始
        :return:
        """
        if len(self.SwathSettings) == 0:
            swath = []
            # 从一个完整cycle开始，此处idx表示的原始scan idx，不是过滤scan后的idx
            while True:
                idx = np.random.choice(len(self.raw_ms1_idx) - 2)
                if self.raw_ms1_idx[idx+1] - self.raw_ms1_idx[idx] == self.windows_num + 1:
                    break

            idx_start = self.raw_ms1_idx[idx] + 1 # 从ms2开始
            idx_end = self.raw_ms1_idx[idx+1]

            # 遍历一个cycle，获取swath窗口设置
            while idx_start < idx_end:
                swath_windom = self.get_current_scan_window(idx_start)
                if swath_windom not in swath:
                    swath.append(swath_windom)
                    idx_start += 1
            self.swath_pair = swath
            swath = np.array([_ for item in swath for _ in item])

            # 还需要重合overlap
            result = []
            if np.min(np.diff(swath)) < 0:  # 有负即为overlap
                result.append(swath[0])
                idx = 1
                while idx + 1 < len(swath) - 1:
                    result.append(np.mean((swath[idx], swath[idx + 1])))
                    idx += 2
                result.append(swath[-1])
                self.SwathSettings = np.array(result)
            elif np.min(np.diff(swath)) == 0:  # 不存在overlap
                self.SwathSettings = np.sort(np.unique(swath))

    @profile
    def get_ms2_xics_by_fg_mz(self, idx_start, idx_end, ms2_win_idx, mz_query, ppm_tolerance=20.):

        num_windows = len(self.SwathSettings) - 1

        # check for cycle
        result_xics, result_rts = [], []
        for cycle_idx, scan_idx in enumerate(range(idx_start, idx_end+1, num_windows+1)):
            result_rts.append(self.all_rt[scan_idx])
            peaks_mz = self.all_mz[scan_idx + ms2_win_idx]
            peaks_int = self.all_intensity[scan_idx + ms2_win_idx]
            xic_v_6 = sj_find_ok_matches(peaks_mz, peaks_int, mz_query, ppm_tolerance)
            result_xics.append(xic_v_6)

        result_xics = np.array(result_xics).T
        result_rts = np.array(result_rts)

        return result_xics, result_rts

def load_ms(ms_file, type):
    start_time = time.time()
    ms = mz_Reader(ms_file, type)

    print('{:<30}{:.2f}s'.format('ms loading time', time.time() - start_time))

    return ms
