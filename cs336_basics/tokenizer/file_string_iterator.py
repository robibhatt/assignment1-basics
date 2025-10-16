from typing import BinaryIO

EMPTY_BYTES = ''.encode('utf-8')

class FileStringIterator:

    def __init__(self, 
                 file: BinaryIO,
                 start: int,
                 end: int,
                 chunk_size: int):
        
        self.file = file
        self.start = start
        self.end = end
        self.chunk_size = chunk_size
        self.prefix= EMPTY_BYTES

    def __iter__(self):
        return self
    
    def __next__(self):

        # if we are done with the file, we wash our hands clean
        if self.start == self.end:
            assert (self.prefix == EMPTY_BYTES) # in a valid file we should never be left with undecodable stuff
            raise StopIteration
        
        # first get the next set of bytes
        self.file.seek(self.start)
        next_chunk_size = min(self.end - self.start, self.chunk_size)
        self.start += next_chunk_size # update the start
        data = self.file.read(next_chunk_size)
        data = self.prefix+data # include old prefix of unreadable crap. 
        self.prefix = EMPTY_BYTES # reset the prefix as it is now included in data we are processing

        # decode the string after including our prefix
        decoded_data : str = data.decode('utf-8', errors='surrogateescape')

        # get the last bytes that didn't decode properly
        last_index = len(decoded_data)-1
        last_char_bytes_int = ord(decoded_data[last_index])
        while last_index >= 0 and (0xDC80 <= last_char_bytes_int <= 0xDCFF):
            self.prefix = bytes([last_char_bytes_int - 0xDC00]) + self.prefix # update the prefix
            last_index -= 1
            last_char_bytes_int = ord(decoded_data[last_index])

        # return the string that we could decode
        return decoded_data[:last_index+1]
    


